from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_model(num_classes: int):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_preprocessing(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def load_approved_model(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Approved model checkpoint not found: {checkpoint_path}"
        )

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if checkpoint.get("architecture") != "MobileNetV3-Small":
        raise ValueError("Unexpected model architecture.")

    class_names = checkpoint.get("class_names")
    class_to_idx = checkpoint.get("class_to_idx")

    if not class_names or not class_to_idx:
        raise ValueError("Checkpoint class mapping is missing.")

    if len(class_names) != len(class_to_idx):
        raise ValueError("Invalid checkpoint class mapping.")

    model = create_model(len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, device


def preprocess_image(image, image_size=224):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    image = image.convert("RGB")

    tensor = get_preprocessing(image_size)(image)

    if tensor.shape != (3, image_size, image_size):
        raise ValueError("Unexpected preprocessed image shape.")

    if not torch.isfinite(tensor).all():
        raise ValueError("Preprocessed image contains invalid values.")

    return tensor


@torch.inference_mode()
def predict_image(image, model, checkpoint, device):
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image.")

    image_size = int(checkpoint.get("image_size", 224))
    tensor = preprocess_image(image, image_size)

    batch = tensor.unsqueeze(0).to(device)
    logits = model(batch)

    if not torch.isfinite(logits).all():
        raise ValueError("Model produced invalid output.")

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = probabilities.max(dim=1)

    predicted_index = int(predicted_index.item())
    confidence = float(confidence.item())

    class_names = checkpoint["class_names"]

    return {
        "predicted_class": class_names[predicted_index],
        "predicted_class_id": predicted_index,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(probabilities[0, i].item())
            for i in range(len(class_names))
        },
        "model_version": checkpoint.get("model_version", "unknown")
    }
