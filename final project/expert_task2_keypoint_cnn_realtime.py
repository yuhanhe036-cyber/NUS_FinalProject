"""Expert Task 2: real-time effects using the MediaPipe keypoint CNN."""

from pathlib import Path
import time

import cv2
import numpy as np
import torch

from expert_keypoint_cnn import (
    FEATURE_VERSION,
    KeypointExpressionCNN,
    LANDMARK_REGION_SLICES,
    REGION_CONV_LAYERS,
    create_face_landmarker,
    image_from_bgr,
    landmarks_to_pixels,
    result_to_keypoints,
    standardize_keypoints,
)


BASE_DIR = Path(__file__).resolve().parent
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
MODEL_PATH = BASE_DIR / "expert_task1_keypoint_cnn_5conv.pt"
CAMERA_INDEX = 0
NEUTRAL_CALIBRATION_FRAMES = 45


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find model: {MODEL_PATH}\n"
            "Run expert_task1_keypoint_cnn_train.py first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(MODEL_PATH, map_location=device)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model and keypoint extractor versions do not match.")
    if payload.get("region_conv_layers") != REGION_CONV_LAYERS:
        raise ValueError("The model is not the expected five-layer region CNN.")
    labels = payload["labels"]
    model = KeypointExpressionCNN(len(labels)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    if "neutral_reference" not in payload:
        raise ValueError(
            "The model has no neutral reference. Run the updated "
            "expert_task1_keypoint_cnn_train.py."
        )
    neutral_reference = np.asarray(
        payload["neutral_reference"], dtype=np.float32
    ).reshape(-1, 3)
    return model, labels, feature_mean, feature_std, neutral_reference, device


def draw_star(frame, center, size, color):
    cx, cy = center
    points = [
        (cx, cy - size), (cx + size // 4, cy - size // 4),
        (cx + size, cy - size // 4), (cx + size // 2, cy + size // 6),
        (cx + size * 2 // 3, cy + size), (cx, cy + size // 2),
        (cx - size * 2 // 3, cy + size), (cx - size // 2, cy + size // 6),
        (cx - size, cy - size // 4), (cx - size // 4, cy - size // 4),
    ]
    cv2.fillPoly(frame, [np.asarray(points, dtype=np.int32)], color)


def draw_keypoints(frame, pixels):
    region_colors = {
        "jaw": (150, 60, 20),
        "eyebrows": (180, 0, 180),
        "nose": (0, 130, 255),
        "eyes": (255, 120, 0),
        "mouth": (0, 180, 0),
    }
    for point_index, (x, y) in enumerate(pixels):
        point_color = (0, 0, 255)
        for region_name, region_slice in LANDMARK_REGION_SLICES.items():
            if region_slice.start <= point_index < region_slice.stop:
                point_color = region_colors[region_name]
                break
        cv2.circle(frame, (x, y), 2, point_color, -1)


def draw_effect(frame, label, pixels):
    if not pixels:
        return
    x_values, y_values = zip(*pixels)
    x1, y1 = max(0, min(x_values)), max(0, min(y_values))
    x2, y2 = min(frame.shape[1] - 1, max(x_values)), min(
        frame.shape[0] - 1, max(y_values)
    )
    colors = {
        "angry": (0, 0, 255),
        "disgust": (0, 160, 0),
        "fear": (180, 0, 180),
        "happy": (0, 220, 255),
        "neutral": (180, 180, 180),
        "sad": (255, 120, 0),
        "surprise": (255, 180, 0),
    }
    color = colors[label]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        label.capitalize(),
        (x1, max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    draw_keypoints(frame, pixels)
    if label == "happy":
        draw_star(frame, (x1, max(25, y1 - 25)), 14, color)
        draw_star(frame, (x2, max(25, y1 - 20)), 12, color)
    elif label == "surprise":
        cv2.circle(frame, ((x1 + x2) // 2, max(25, y1 - 25)), 14, color, 2)


def main():
    (
        model,
        labels,
        feature_mean,
        feature_std,
        neutral_reference,
        device,
    ) = load_model()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {CAMERA_INDEX}")

    with create_face_landmarker(LANDMARKER_MODEL_PATH, "VIDEO") as landmarker:
        calibration_samples = []
        neutral_offset = None
        print(f"Using device: {device}.")
        print(
            f"Hold a neutral expression for {NEUTRAL_CALIBRATION_FRAMES} valid "
            "frames. Press c to recalibrate or q to quit."
        )
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(image_from_bgr(frame), timestamp_ms)
            keypoints = result_to_keypoints(result)
            label = None
            confidence = None
            classifier_ms = None
            status = "No face detected"

            if keypoints is not None:
                pixels = landmarks_to_pixels(result, frame.shape)
                if neutral_offset is None:
                    calibration_samples.append(keypoints.copy())
                    draw_keypoints(frame, pixels)
                    status = (
                        "Neutral calibration: "
                        f"{len(calibration_samples)}/{NEUTRAL_CALIBRATION_FRAMES}"
                    )
                    if len(calibration_samples) >= NEUTRAL_CALIBRATION_FRAMES:
                        personal_neutral = np.median(
                            np.stack(calibration_samples), axis=0
                        )
                        neutral_offset = neutral_reference - personal_neutral
                        calibration_samples.clear()
                        print("Neutral geometry calibration completed.")
                else:
                    classifier_start = time.perf_counter()
                    calibrated_keypoints = keypoints + neutral_offset
                    tensor = standardize_keypoints(
                        calibrated_keypoints, feature_mean, feature_std
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        probabilities = torch.softmax(model(tensor.to(device)), dim=1)
                        confidence_tensor, index_tensor = probabilities.max(dim=1)
                    predicted_index = int(index_tensor.item())
                    label = labels[predicted_index]
                    confidence = float(confidence_tensor.item())
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    classifier_ms = (
                        time.perf_counter() - classifier_start
                    ) * 1000.0
                    draw_effect(frame, label, pixels)
                    status = f"Expression: {label} ({confidence:.2f})"

            cv2.putText(
                frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2
            )
            classifier_text = (
                f"Prediction time: {classifier_ms:.2f} ms | Required: <30 ms"
                if classifier_ms is not None
                else "Prediction time: -- ms | Required: <30 ms"
            )
            classifier_color = (
                (0, 220, 0)
                if classifier_ms is not None and classifier_ms < 30.0
                else (0, 0, 255)
                if classifier_ms is not None
                else (255, 255, 255)
            )
            cv2.putText(
                frame,
                classifier_text,
                (20, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                classifier_color,
                2,
            )
            cv2.imshow("Expert Task 2 - Keypoint CNN", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                calibration_samples.clear()
                neutral_offset = None
                print("Neutral geometry recalibration started.")
            elif key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
