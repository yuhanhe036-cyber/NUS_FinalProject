"""Generate a montage of real test-set errors from the five-layer keypoint CNN."""

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from expert_keypoint_cnn import FEATURE_VERSION, KeypointExpressionCNN, NUM_KEYPOINTS
from expert_task1_keypoint_cnn_train import (
    LABELS,
    LANDMARKER_MODEL_PATH,
    extract_keypoint,
    image_paths,
)
from expert_mediapipe_68_svm import create_face_landmarker


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "expert_task1_keypoint_cnn_5conv.pt"
CACHE_PATH = BASE_DIR / "expert_task1_keypoint_cnn_features.npz"
OUTPUT_PATH = BASE_DIR / "expert_task1_cnn_failure_examples.png"
CSV_PATH = BASE_DIR / "expert_task1_cnn_failure_examples.csv"
SEARCH_LIMIT_PER_CLASS = 250


def load_model_and_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(MODEL_PATH, map_location=device)
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError("Model and cache feature versions do not match.")

    model = KeypointExpressionCNN(len(LABELS)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)

    cache = np.load(CACHE_PATH)
    X_test = cache["X_test"].astype(np.float32)
    y_test = cache["y_test"].astype(np.int64)
    standardized = (X_test - feature_mean) / feature_std
    tensors = torch.from_numpy(
        standardized.reshape(-1, NUM_KEYPOINTS, 3).transpose(0, 2, 1).copy()
    )

    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(tensors), 256):
            batch = tensors[start : start + 256].to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    return y_test, predictions, confidences


def select_failure_indices(y_true, predictions, confidences):
    selected = []
    for class_index, label in enumerate(LABELS):
        class_indices = np.flatnonzero(y_true == class_index)
        candidates = class_indices[:SEARCH_LIMIT_PER_CLASS]
        wrong = candidates[predictions[candidates] != class_index]
        if len(wrong) == 0:
            continue
        selected_index = wrong[np.argmax(confidences[wrong])]
        accepted_ordinal = int(np.flatnonzero(class_indices == selected_index)[0])
        selected.append(
            {
                "cache_index": int(selected_index),
                "accepted_ordinal": accepted_ordinal,
                "true_index": class_index,
                "true_label": label,
                "predicted_index": int(predictions[selected_index]),
                "predicted_label": LABELS[int(predictions[selected_index])],
                "confidence": float(confidences[selected_index]),
            }
        )
    return selected


def locate_original_images(selected):
    by_label = {item["true_label"]: item for item in selected}
    with create_face_landmarker(LANDMARKER_MODEL_PATH, "IMAGE") as landmarker:
        for label, item in by_label.items():
            accepted_count = 0
            for path in image_paths("test", label):
                feature, _ = extract_keypoint(path, landmarker)
                if feature is None:
                    continue
                if accepted_count == item["accepted_ordinal"]:
                    item["path"] = path
                    break
                accepted_count += 1
            if "path" not in item:
                raise RuntimeError(f"Could not map cached feature back to {label} image.")
    return selected


def create_montage(selected):
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 7))
    axes = np.asarray(axes).reshape(-1)

    for axis, item in zip(axes, selected):
        image = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(
            f"Ground truth: {item['true_label']}\n"
            f"Predicted: {item['predicted_label']} "
            f"({item['confidence']:.2f})",
            fontsize=11,
            color="darkred",
        )
        axis.axis("off")
    for axis in axes[len(selected) :]:
        axis.axis("off")

    figure.suptitle(
        "Five-layer MediaPipe 68-point CNN: Test-set Failure Cases",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["file", "ground_truth", "predicted", "confidence"],
        )
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "file": item["path"].name,
                    "ground_truth": item["true_label"],
                    "predicted": item["predicted_label"],
                    "confidence": f"{item['confidence']:.6f}",
                }
            )


def main():
    y_true, predictions, confidences = load_model_and_predictions()
    selected = select_failure_indices(y_true, predictions, confidences)
    selected = locate_original_images(selected)
    create_montage(selected)
    for item in selected:
        print(
            f"{item['path'].name}: {item['true_label']} -> "
            f"{item['predicted_label']} ({item['confidence']:.4f})"
        )
    print(f"Saved montage: {OUTPUT_PATH}")
    print(f"Saved details: {CSV_PATH}")


if __name__ == "__main__":
    main()
