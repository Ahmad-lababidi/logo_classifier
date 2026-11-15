# Logo Classifier

A small web app for comparing three logo classifiers: a DenseLogoCNN, a k-nearest-neighbors classifier built on the CNN's features, and a fine-tuned ResNet18. Upload an image in the Streamlit page to see each model's top predictions and confidence scores. The FastAPI service also exposes validation metrics.

## How it fits together

- `src/` contains the models, image preprocessing, training code, and utilities.
- `backend/api.py` serves predictions and metrics through FastAPI.
- `frontend/app.py` provides the Streamlit interface.
- `docker-compose.yaml` runs the API and interface as two services.

Images are resized and normalized before inference. The API accepts PNG, JPEG, WebP, and BMP images up to 10 MB. Prediction requests require an API key and are limited to 20 requests per minute per client.

## Requirements

- Python 3.11
- A logo dataset arranged in class folders
- Docker Compose (optional)

The dataset and trained model files are not stored in this repository. To train the models, put your dataset under `data/1000 Logos - Temp` (one folder per class, with images inside) or pass another path with `--data_dir`.

## Train the models

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
```

Training writes the class list, model weights, kNN classifier, and validation metrics to `saved_models/`. Training can take a while, especially without a GPU. Use `python -m src.train --help` to see options such as epochs, batch size, and dataset path.

## Run locally

Train the models first. Then set an API key in your shell and start the API:

```bash
export LOGO_API_KEY="choose-a-secret-key"
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

In a second terminal, activate the same virtual environment and start the UI:

```bash
export LOGO_API_KEY="choose-a-secret-key"
streamlit run frontend/app.py
```

On Windows PowerShell, set the variable with `$env:LOGO_API_KEY = "choose-a-secret-key"` in each terminal instead. The API defaults to `http://localhost:8000/predict`; set `API_URL` on the frontend if the API is running elsewhere.

Open the Streamlit URL shown in the terminal, usually http://localhost:8501. The API docs are at http://localhost:8000/docs.

## Run with Docker Compose

After training has created the files in `saved_models/`, set `LOGO_API_KEY` in your environment and run:

```bash
docker compose up --build
```

Open http://localhost:8501. The API is available at http://localhost:8000. Keep the API key private; do not commit it or put it in a public file.
