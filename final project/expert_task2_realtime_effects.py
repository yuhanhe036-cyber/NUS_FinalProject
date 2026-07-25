from pathlib import Path

import cv2
import numpy as np
import torch

from expert_expression_cnn import ExpressionCNN, preprocess_gray_face


BASE_DIR = Path(__file__).resolve().parent
HAAR_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = BASE_DIR / "expert_task1_expression_cnn.pt"
CAMERA_INDEX = 0
FACE_PADDING_RATIO = 0.10


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find trained CNN model: {MODEL_PATH}\n"
            "Run expert_task1_train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(MODEL_PATH, map_location=device)
    labels = payload["labels"]
    model = ExpressionCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, labels, device


def load_face_detector():
    detector = cv2.CascadeClassifier(str(HAAR_PATH))
    if detector.empty():
        raise FileNotFoundError(f"Cannot load Haar cascade: {HAAR_PATH}")
    return detector


def choose_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def padded_face_crop(gray, face_rect):
    x, y, w, h = [int(value) for value in face_rect]
    padding_x = int(w * FACE_PADDING_RATIO)
    padding_y = int(h * FACE_PADDING_RATIO)
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(gray.shape[1], x + w + padding_x)
    y2 = min(gray.shape[0], y + h + padding_y)
    return gray[y1:y2, x1:x2]


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


def draw_expression_effect(frame, label, face_rect):
    x, y, w, h = [int(value) for value in face_rect]
    if label == "happy":
        color, text = (0, 220, 255), "Happy"
        draw_star(frame, (x, max(25, y - 25)), 16, color)
        draw_star(frame, (x + w, max(25, y - 20)), 14, color)
    elif label == "surprise":
        color, text = (255, 180, 0), "Surprise"
        cv2.circle(frame, (x + w // 2, max(30, y - 30)), 16, color, 2)
    elif label == "angry":
        color, text = (0, 0, 255), "Angry"
    elif label == "sad":
        color, text = (255, 120, 0), "Sad"
    elif label == "fear":
        color, text = (180, 0, 180), "Fear"
    elif label == "disgust":
        color, text = (0, 160, 0), "Disgust"
    else:
        color, text = (180, 180, 180), "Neutral"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(
        frame, text, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )


def main():
    model, labels, device = load_model()
    face_detector = load_face_detector()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {CAMERA_INDEX}")

    print(f"Using device: {device}. Press q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(80, 80)
        )
        face_rect = choose_largest_face(faces)
        label = None

        if face_rect is not None:
            face_tensor = preprocess_gray_face(padded_face_crop(gray, face_rect))
            if face_tensor is not None:
                with torch.no_grad():
                    predicted_index = int(model(face_tensor.to(device)).argmax(dim=1).item())
                label = labels[predicted_index]
                draw_expression_effect(frame, label, face_rect)

        status = f"Expression: {label}" if label is not None else "No face detected"
        cv2.putText(
            frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
        cv2.putText(
            frame,
            "Expert Task 2: raw CNN prediction | Press q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Expert Task 2 - Realtime CNN Expression Effects", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
