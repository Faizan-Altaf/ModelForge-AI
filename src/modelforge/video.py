from pathlib import Path
import cv2
import pandas as pd

from PIL import Image

from .inference import predict_image


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def calculate_frame_stride(fps, interval_seconds):
    if fps <= 0:
        raise ValueError("Video FPS must be greater than zero.")

    if interval_seconds <= 0:
        raise ValueError("Frame interval must be greater than zero.")

    return max(1, round(fps * interval_seconds))


def process_video(
    input_path,
    output_path,
    model,
    checkpoint,
    device,
    interval_seconds=1.0
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video format.")

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise ValueError("Video file is missing or empty.")

    if interval_seconds <= 0:
        raise ValueError("Interval must be greater than zero.")

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise ValueError("Unable to open uploaded video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Invalid video metadata.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = calculate_frame_stride(fps, interval_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_fps = 1.0 / interval_seconds

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create annotated output video.")

    records = []
    frame_index = 0
    sampled_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % stride == 0:

                timestamp = frame_index / fps

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                pil_image = Image.fromarray(rgb_frame)

                result = predict_image(
                    pil_image,
                    model,
                    checkpoint,
                    device
                )

                label = result["predicted_class"]
                confidence = result["confidence"]

                text = f"{label} | {confidence:.1%}"

                cv2.rectangle(
                    frame,
                    (8, 8),
                    (min(width - 8, 500), 58),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    text,
                    (18, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                writer.write(frame)

                records.append({
                    "sampled_frame": sampled_index,
                    "source_frame": frame_index,
                    "timestamp_seconds": round(timestamp, 3),
                    "predicted_class": label,
                    "confidence": confidence
                })

                sampled_index += 1

            frame_index += 1

    finally:
        capture.release()
        writer.release()

    if not records:
        raise RuntimeError("No video frames could be processed.")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Annotated video output was not created.")

    results_df = pd.DataFrame(records)

    distribution = (
        results_df["predicted_class"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "total_source_frames": frame_count,
        "sampled_frames": len(results_df),
        "source_fps": fps,
        "video_duration_seconds": frame_count / fps,
        "frame_interval_seconds": interval_seconds,
        "majority_predicted_class": results_df[
            "predicted_class"
        ].value_counts().idxmax(),
        "mean_confidence": float(
            results_df["confidence"].mean()
        ),
        "class_distribution": distribution
    }

    return results_df, summary
