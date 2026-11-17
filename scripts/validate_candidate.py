from pathlib import Path
import argparse
import json
import math
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from modelforge.inference import create_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--candidate",
    default="models/candidates/challenger_candidate.pth"
)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]

active_path = root / "models/champion/champion.pth"
candidate_path = root / args.candidate

if not active_path.exists():
    raise FileNotFoundError(active_path)

if not candidate_path.exists():
    raise FileNotFoundError(candidate_path)

active = torch.load(active_path, map_location="cpu", weights_only=False)
candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)

active_acc = float(active["validation_accuracy"])
candidate_acc = float(candidate["validation_accuracy"])

active_f1 = float(active["validation_macro_f1"])
candidate_f1 = float(candidate["validation_macro_f1"])

accuracy_gain = candidate_acc - active_acc
f1_change = candidate_f1 - active_f1

checks = {
    "finite_metrics": all(
        math.isfinite(x)
        for x in [active_acc, candidate_acc, active_f1, candidate_f1]
    ),
    "accuracy_improvement": accuracy_gain >= 0.005,
    "macro_f1_requirement": f1_change >= -0.01,
    "same_dataset": (
        active["dataset_fingerprint"]
        == candidate["dataset_fingerprint"]
    ),
    "same_class_names": (
        list(active["class_names"])
        == list(candidate["class_names"])
    ),
    "same_class_mapping": (
        active["class_to_idx"]
        == candidate["class_to_idx"]
    )
}

smoke_test = False

try:
    model = create_model(int(candidate["num_classes"]))
    model.load_state_dict(candidate["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))

    smoke_test = (
        output.shape == (1, int(candidate["num_classes"]))
        and torch.isfinite(output).all().item()
    )

except Exception as error:
    print("Smoke test failed:", error)

checks["checkpoint_smoke_test"] = smoke_test

result = {
    "candidate": str(candidate_path),
    "baseline": str(active_path),
    "accuracy_gain": accuracy_gain,
    "macro_f1_change": f1_change,
    "checks": checks,
    "decision": "ELIGIBLE_FOR_REVIEW" if all(checks.values()) else "REJECTED"
}

output_path = root / "candidate_validation_report.json"

output_path.write_text(
    json.dumps(result, indent=4),
    encoding="utf-8"
)

print(json.dumps(result, indent=4))

if not all(checks.values()):
    raise SystemExit("Candidate failed one or more validation gates.")

print("Candidate passed automated validation. Manual promotion review is still required.")
