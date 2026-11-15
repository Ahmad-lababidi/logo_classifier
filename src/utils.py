import os
import json
import time
import logging
import hmac
from collections import defaultdict
import torch
from fastapi import HTTPException, Security, Request, status
from fastapi.security.api_key import APIKeyHeader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LogoClassifier")

def save_checkpoint(model: torch.nn.Module, path: str, metadata: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )
    logger.info(f"Saved model weights to {path}")

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded model weights from {path}")
    return model

def save_class_names(class_names: list, path: str = "saved_models/class_names.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)

def load_class_names(path: str = "saved_models/class_names.json") -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class names file not found at {path}. Train models first.")
    with open(path, "r") as f:
        return json.load(f)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = os.getenv("LOGO_API_KEY")
    if not expected_api_key:
        logger.error("LOGO_API_KEY is not configured; refusing API authentication.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )
    if not api_key or not hmac.compare_digest(api_key, expected_api_key):
        logger.warning("Unauthorized API access attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header."
        )
    return api_key

class RateLimiter:
    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.client_requests = defaultdict(list)

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        self.client_requests[client_ip] = [
            ts for ts in self.client_requests[client_ip]
            if now - ts < self.window_seconds
        ]
        if len(self.client_requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s."
            )
        self.client_requests[client_ip].append(now)
