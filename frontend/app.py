import os
from urllib.parse import urlsplit, urlunsplit

import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Logo Classifier", layout="wide")
st.title("Logo Classifier")
st.write("Compare a scratch CNN, its kNN classifier, and a fine-tuned ResNet18.")

top_k = 3

api_url = os.getenv("API_URL", "http://localhost:8000/predict")
api_key = os.getenv("LOGO_API_KEY", "")
parsed_api_url = urlsplit(api_url)
metrics_url = urlunsplit((
    parsed_api_url.scheme,
    parsed_api_url.netloc,
    parsed_api_url.path.rsplit("/", 1)[0] + "/metrics",
    "",
    "",
))

st.header("Validation metrics")
if not api_key:
    st.info("Metrics are unavailable because API authentication is not configured.")
else:
    try:
        metrics_response = requests.get(
            metrics_url, headers={"X-API-Key": api_key}, timeout=5
        )
        if metrics_response.ok:
            metrics = metrics_response.json()
            model_metrics = metrics.get("models", {})
            rows = []
            for name, label in (
                ("scratch_cnn", "DenseLogoCNN"),
                ("scratch_knn", "DenseLogoCNN + kNN"),
                ("resnet", "ResNet18"),
            ):
                values = model_metrics.get(name, {})
                accuracy = values.get("accuracy")
                macro_f1 = values.get("macro_f1")
                rows.append({
                    "Model": label,
                    "Accuracy": f"{accuracy:.1%}" if accuracy is not None else "Unavailable",
                    "Macro F1": f"{macro_f1:.1%}" if macro_f1 is not None else "Unavailable",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption(
                f"Validation set: {metrics.get('validation_images', 'unknown')} images; "
                f"{metrics.get('class_count', 'unknown')} classes; seed {metrics.get('seed', 'unknown')}"
            )
        else:
            st.info("Metrics are unavailable. Train the models to generate them.")
    except requests.exceptions.RequestException:
        st.info("Metrics are unavailable while the API is offline.")

st.divider()
st.header("Compare a logo")
uploaded_file = st.file_uploader(
    "Upload a logo image",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
)

if uploaded_file is not None:
    image_col, results_col = st.columns([1, 3])
    with image_col:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, width=260)
        except Exception:
            st.error("The selected file is not a valid image.")

    with results_col:
        if st.button("Classify logo", type="primary", use_container_width=True):
            if not api_key:
                st.error("API authentication is not configured on the server.")
            else:
                try:
                    response = requests.post(
                        api_url,
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "image/png",
                            )
                        },
                        headers={"X-API-Key": api_key},
                        params={"model_type": "all", "top_k": top_k},
                        timeout=30,
                    )
                    if not response.ok:
                        detail = response.json().get("detail", "Request failed.")
                        st.error(f"API error ({response.status_code}): {detail}")
                    else:
                        results = response.json().get("results", {})
                        columns = st.columns(3)
                        for column, title, key in (
                            (columns[0], "DenseLogoCNN", "scratch_cnn"),
                            (columns[1], "DenseLogoCNN + kNN", "scratch_knn"),
                            (columns[2], "ResNet18", "resnet_finetuned"),
                        ):
                            with column:
                                st.subheader(title)
                                result = results.get(key, {})
                                predictions = result.get("predictions", [])
                                if not predictions:
                                    st.warning("No predictions were returned.")
                                    continue
                                first = predictions[0]
                                st.metric(
                                    "Top prediction",
                                    first["label"],
                                    f"{first['confidence']:.1%} confidence",
                                )
                                st.caption(f"Inference time: {result.get('inference_time_ms')} ms")
                                for prediction in predictions:
                                    confidence = max(0.0, min(1.0, float(prediction["confidence"])))
                                    st.progress(
                                        confidence,
                                        text=f"{prediction['label']} ({confidence:.1%})",
                                    )
                except requests.exceptions.RequestException:
                    st.error("Could not connect to the FastAPI backend. Check that the backend is running.")
                except (ValueError, KeyError, TypeError):
                    st.error("The API returned an unexpected response.")
