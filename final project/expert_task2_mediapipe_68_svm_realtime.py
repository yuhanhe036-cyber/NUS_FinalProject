"""Expert Task 2: real-time effects from the standalone MediaPipe 68-point SVM."""

from pathlib import Path
import pickle
import time

import cv2
import numpy as np

from expert_mediapipe_68_svm import (
    FEATURE_VERSION,
    LANDMARK_REGION_SLICES,
    create_face_landmarker,
    image_from_bgr,
    landmarks_to_pixels,
    result_to_feature,
)


BASE_DIR = Path(__file__).resolve().parent
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
MODEL_PATH = BASE_DIR / "expert_task1_mediapipe_68_expression_svm.pkl"
CAMERA_INDEX = 0


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find SVM model: {MODEL_PATH}\n"
            "Run expert_task1_mediapipe_68_svm_train.py first."
        )
    with open(MODEL_PATH, "rb") as input_file:
        payload = pickle.load(input_file)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("SVM model and MediaPipe feature versions do not match.")
    return payload["model"], payload["labels"]


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
    model, labels = load_model()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {CAMERA_INDEX}")

    with create_face_landmarker(LANDMARKER_MODEL_PATH, "VIDEO") as landmarker:
        print("MediaPipe 68-point + raw SVM prediction. Press q to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(image_from_bgr(frame), timestamp_ms)
            feature = result_to_feature(result)
            label = None
            classifier_ms = None
            if feature is not None:
                classifier_start = time.perf_counter()
                predicted_index = int(model.predict(feature.reshape(1, -1))[0])
                classifier_ms = (time.perf_counter() - classifier_start) * 1000.0
                label = labels[predicted_index]
                draw_effect(frame, label, landmarks_to_pixels(result, frame.shape))

            status = f"Expression: {label}" if label is not None else "No face detected"
            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
            )
            classifier_text = (
                f"Prediction time: {classifier_ms:.2f} ms | Required: <30 ms"
                if classifier_ms is not None
                else "Prediction time: -- ms | Required: <30 ms"
            )
            cv2.putText(
                frame,
                classifier_text,
                (20, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 220, 0)
                if classifier_ms is not None and classifier_ms < 30.0
                else (0, 0, 255)
                if classifier_ms is not None
                else (255, 255, 255),
                2,
            )
            cv2.imshow("Expert Task 2 - MediaPipe 68 SVM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
