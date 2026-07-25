"""MediaPipe Tasks Face Landmarker helpers shared by the SVM training and demo."""

from pathlib import Path
import shutil
from urllib.request import urlopen

import mediapipe as mp
import numpy as np


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Expression-relevant contours: eyebrows, eyes, nose and both lip contours.
FEATURE_LANDMARK_INDICES = np.array(
    [
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        1, 2, 4, 5, 6, 19, 94, 97, 168, 197,
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    ],
    dtype=np.int32,
)
LEFT_EYE_INDICES = np.array([33, 160, 158, 133, 153, 144], dtype=np.int32)
RIGHT_EYE_INDICES = np.array([362, 385, 387, 263, 373, 380], dtype=np.int32)
FEATURE_VERSION = "mediapipe_3d_expression_regions_v1"


def ensure_face_landmarker_model(model_path):
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".part")
    print(f"Downloading MediaPipe Face Landmarker model: {model_path.name}")
    try:
        with urlopen(FACE_LANDMARKER_MODEL_URL, timeout=60) as response, open(
            temporary_path, "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return model_path


def create_face_landmarker(model_path, running_mode="IMAGE"):
    model_path = ensure_face_landmarker_model(model_path)
    vision = mp.tasks.vision
    mode = vision.RunningMode.IMAGE if running_mode == "IMAGE" else vision.RunningMode.VIDEO
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def image_from_bgr(frame):
    rgb = frame[:, :, ::-1].copy()
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def result_to_feature(result):
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    if len(landmarks) <= int(FEATURE_LANDMARK_INDICES.max()):
        return None

    points = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
    left_eye_center = points[LEFT_EYE_INDICES].mean(axis=0)
    right_eye_center = points[RIGHT_EYE_INDICES].mean(axis=0)
    eye_distance = float(np.linalg.norm(right_eye_center - left_eye_center))
    if eye_distance <= 1e-6:
        return None

    normalized = (points[FEATURE_LANDMARK_INDICES] - (left_eye_center + right_eye_center) / 2.0)
    return (normalized / eye_distance).flatten().astype(np.float32)


def landmark_pixels(result, frame_shape):
    if not result.face_landmarks:
        return None
    height, width = frame_shape[:2]
    return [
        (int(point.x * width), int(point.y * height))
        for point in result.face_landmarks[0]
    ]
