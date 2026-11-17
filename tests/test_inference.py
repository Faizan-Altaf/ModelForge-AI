import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modelforge.inference import (
    preprocess_image,
    create_model,
    load_approved_model,
    predict_image
)


def test_preprocessing_output_shape():

    image = Image.new("RGB", (320, 240), color="white")

    tensor = preprocess_image(image)

    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()


def test_preprocessing_converts_rgb():

    image = Image.new("L", (200, 200), color=128)

    tensor = preprocess_image(image)

    assert tensor.shape[0] == 3


def test_invalid_image_input():

    with pytest.raises(TypeError):
        preprocess_image("not-an-image")


def test_model_output_shape():

    model = create_model(4)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()


def test_approved_checkpoint_integration():

    checkpoint_path = (
        ROOT / "models" / "champion" / "champion.pth"
    )

    if not checkpoint_path.exists():
        pytest.skip("Approved checkpoint not present in checkout.")

    model, checkpoint, device = load_approved_model(checkpoint_path)

    assert checkpoint["model_version"] == "challenger_v3"
    assert len(checkpoint["class_names"]) == 4

    image = Image.new("RGB", (224, 224), color="white")

    result = predict_image(
        image,
        model,
        checkpoint,
        device
    )

    assert result["predicted_class"] in checkpoint["class_names"]
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 4
