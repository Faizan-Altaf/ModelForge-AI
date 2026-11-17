from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "champion" / "champion.pth"
OUTPUT_PATH = ROOT / "models" / "champion" / "release_manifest.json"


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

import torch

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

manifest = {
    "release_version": checkpoint.get("model_version"),
    "architecture": checkpoint.get("architecture"),
    "class_names": checkpoint.get("class_names"),
    "dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
    "training_run_id": checkpoint.get("training_run_id"),
    "sha256": sha256_file(MODEL_PATH),
    "model_file": MODEL_PATH.name,
    "created_at_utc": datetime.now(timezone.utc).isoformat()
}

OUTPUT_PATH.write_text(
    json.dumps(manifest, indent=4),
    encoding="utf-8"
)

print(json.dumps(manifest, indent=4))
print("Manifest saved:", OUTPUT_PATH)
