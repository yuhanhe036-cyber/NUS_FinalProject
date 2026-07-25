from pathlib import Path
import pickle
import time

import cv2

from expert_mediapipe_landmarks import (
    create_face_landmarker,
    image_from_bgr,
    landmark_pixels,
    result_to_feature,
)


BASE_DIR = Path(__file__).resolve().parent
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
MODEL_PATH = BASE_DIR / "expert_task1_mediapipe_svm.pkl"
CAMERA_INDEX = 0


def draw_effect(frame, label, pixels):
    if not pixels:
        return
    x_values, y_values = zip(*pixels)
    x1, y1 = min(x_values), min(y_values)
    x2, y2 = max(x_values), max(y_values)
    colors = {
        "angry": (0, 0, 255), "disgust": (0, 160, 0), "fear": (180, 0, 180),
        "happy": (0, 220, 255), "neutral": (180, 180, 180), "sad": (255, 120, 0),
        "surprise": (255, 180, 0),
    }
    color = colors[label]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, label, (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for x, y in pixels:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run expert_task1_mediapipe_svm_train.py first.")
    with open(MODEL_PATH, "rb") as input_file:
        payload = pickle.load(input_file)
    model, labels = payload["model"], payload["labels"]
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {CAMERA_INDEX}")

    with create_face_landmarker(LANDMARKER_MODEL_PATH, "VIDEO") as landmarker:
        print("Press q to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(image_from_bgr(frame), timestamp_ms)
            feature = result_to_feature(result)
            label = None
            if feature is not None:
                label_index = int(model.predict(feature.reshape(1, -1))[0])
                label = labels[label_index]
                draw_effect(frame, label, landmark_pixels(result, frame.shape))

            status = f"Expression: {label}" if label else "No face detected"
            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "MediaPipe landmarks + raw SVM prediction | Press q to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Expert Task 2 - MediaPipe SVM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
