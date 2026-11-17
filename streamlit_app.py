
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from modelforge.inference import load_approved_model, predict_image

MODEL_PATH = Path(
    os.getenv(
        "MODEL_CHECKPOINT",
        str(ROOT / "models" / "champion" / "champion.pth")
    )
)

DEVICE = "cpu"

# Final untouched test-set metrics
MODEL_ACCURACY = 69.55
MODEL_MACRO_F1 = 69.04

st.set_page_config(
    page_title="ModelForge AI | Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# ============================================================
# SIMPLE STYLING
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 750;
        color: #172b4d;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .summary-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
    }

    .result-class {
        font-size: 27px;
        font-weight: 750;
        color: #087f5b;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def get_model():
    model, checkpoint, device = load_approved_model(
        MODEL_PATH,
        device=DEVICE
    )
    return model, checkpoint, device


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">♻️ ModelForge AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered waste classification using MobileNetV3-Small'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# UPLOAD + IMAGE PREVIEW
# ============================================================

upload_col, preview_col = st.columns([1, 1.3], gap="large")

with upload_col:
    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a waste image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a can, paper, plastic bag, or plastic bottle."
    )

    st.caption("Supported formats: JPG, JPEG, PNG")

with preview_col:
    st.subheader("Image Preview")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
    else:
        st.info("Your uploaded image preview will appear here.")


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    st.divider()

    try:
        with st.spinner("Analyzing image..."):

            model, checkpoint, device = get_model()

            result = predict_image(
                image,
                model,
                checkpoint,
                device
            )

        predicted_class = result["predicted_class"]
        confidence = float(result["confidence"])
        probabilities = result["probabilities"]
        model_version = result.get("model_version", "Champion Model")

        # ----------------------------------------------------
        # FULL SUMMARY
        # ----------------------------------------------------

        st.subheader("Prediction Summary")

        st.markdown(
            f"""
            <div class="summary-card">
                <div style="color:#64748b;font-size:14px;">
                    Predicted Waste Category
                </div>
                <div class="result-class">
                    {predicted_class}
                </div>
                <p style="color:#475569;margin-top:10px;">
                    The model has classified the uploaded image as
                    <b>{predicted_class}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # ACCURACY + CONFIDENCE
        # ----------------------------------------------------

        st.subheader("Model Performance")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Model Test Accuracy",
                f"{MODEL_ACCURACY:.2f}%"
            )

        with metric_col2:
            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

        with metric_col3:
            st.metric(
                "Test Macro F1-Score",
                f"{MODEL_MACRO_F1:.2f}%"
            )

        st.caption(
            "Model accuracy and Macro F1 are measured on the final "
            "held-out test dataset. Prediction confidence is specific "
            "to this uploaded image; it is not the model's accuracy."
        )

        # ----------------------------------------------------
        # CLASS COMPARISON TABLE
        # ----------------------------------------------------

        st.subheader("Class Comparison")

        comparison_data = []

        sorted_probabilities = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for rank, (class_name, probability) in enumerate(
            sorted_probabilities,
            start=1
        ):
            comparison_data.append({
                "Rank": rank,
                "Waste Category": class_name,
                "Confidence": f"{float(probability) * 100:.2f}%",
                "Probability": float(probability),
                "Prediction": (
                    "Predicted Class"
                    if class_name == predicted_class
                    else "Alternative"
                )
            })

        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(
            comparison_df.drop(columns=["Probability"]),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CONFIDENCE BREAKDOWN
        # ----------------------------------------------------

        st.subheader("Confidence Breakdown")

        for class_name, probability in sorted_probabilities:
            probability = float(probability)

            st.write(
                f"**{class_name}** — {probability * 100:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, probability))
            )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.divider()

        st.caption(f"Model: {model_version}")
        st.caption("Architecture: MobileNetV3-Small")
        st.caption("Input Image Size: 224 × 224")

    except Exception as error:
        st.error(f"Prediction failed: {error}")

else:
    st.info("Upload an image to view its prediction and full analysis.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ModelForge AI | Champion Model | Waste Classification"
)
