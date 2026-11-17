# ModelForge AI

**Champion-Challenger Transfer Learning with MLflow and GitHub Actions**

ModelForge AI is a waste image and sampled-video classification system built
with MobileNetV3-Small, PyTorch, MLflow and Streamlit.

## Supported classes

- Can
- Paper
- Plastic Bag
- Plastic Bottle

## Model

The approved model is Challenger V3, promoted after validation-based
Champion-Challenger comparison.

Architecture: MobileNetV3-Small

Input size: 224 x 224

Preprocessing: RGB, ImageNet normalization

The model predicts one class for an input image.

## Application features

- Single-image classification
- Video upload and sampled-frame classification
- Frame-level prediction timeline
- Confidence scores
- Class distribution
- Annotated sampled-frame video download
- CSV prediction report download

Video processing is frame-level classification. The application does not
perform object localization or bounding-box detection.

## Project structure

```text
ModelForge-AI/
├── .github/workflows/
│   ├── ci.yml
│   └── model-release.yml
├── .streamlit/
│   └── config.toml
├── models/
│   └── champion/
│       └── champion.pth
├── scripts/
│   ├── create_release_manifest.py
│   └── rollback_champion.py
├── src/modelforge/
│   ├── __init__.py
│   ├── inference.py
│   └── video.py
├── tests/
│   ├── conftest.py
│   ├── test_inference.py
│   └── test_video.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Local setup

```powershell
cd E:\ModelForge-AI

python -m pip install -r requirements.txt

python -m pytest -v tests

python -m streamlit run streamlit_app.py
```

## MLflow

The training experiments, Champion-Challenger comparisons, promotion decision,
test evaluation and inference traces are tracked in MLflow.

## CI/CD

GitHub Actions runs automated tests on pushes and pull requests.

Pushing a version tag such as `v1.0.0` triggers model release validation,
SHA-256 manifest generation and GitHub Release creation.

## Model integrity

Generate the model manifest:

```powershell
python scripts/create_release_manifest.py
```

## Rollback

Restore the latest available Champion backup:

```powershell
python scripts/rollback_champion.py
```

Restore a particular backup:

```powershell
python scripts/rollback_champion.py models/champion/champion_backup_YYYYMMDD_HHMMSS.pth
```

## Dataset

The dataset is derived from the existing waste detection dataset.
The original train, validation and test split assignments were preserved.

Document the original public dataset source and its license before publication.

## Limitations

This is an image classifier, not an object detector. For video, it classifies
sampled frames independently. Small or multiple objects in a frame may reduce
prediction reliability.
