"""Standalone MediaPipe 68-point helpers for the Expert-level SVM version."""

from pathlib import Path
import shutil
from urllib.request import urlopen

import mediapipe as mp
import numpy as np


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Conventional 68-point anatomy mapped from MediaPipe Face Landmarker:
# jaw 17, eyebrows 10, nose 9, eyes 12 and mouth 20.
FEATURE_LANDMARK_INDICES = np.asarray(
    [
        234, 93, 132, 58, 172, 136, 150, 176, 152,
        400, 378, 379, 365, 397, 288, 361, 454,
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        168, 6, 197, 195, 98, 97, 2, 326, 327,
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        61, 185, 39, 0, 269, 409, 291, 375, 405, 17, 181, 146,
        78, 81, 13, 311, 308, 402, 14, 178,
    ],
    dtype=np.int32,
)
LEFT_EYE_INDICES = np.asarray([33, 160, 158, 133, 153, 144], dtype=np.int32)
RIGHT_EYE_INDICES = np.asarray([362, 385, 387, 263, 373, 380], dtype=np.int32)
NUM_KEYPOINTS = len(FEATURE_LANDMARK_INDICES)
FEATURE_DIMENSION = NUM_KEYPOINTS * 3
FEATURE_VERSION = "mediapipe_anatomical_68x3_eye_aligned_v3"
LANDMARK_REGION_SLICES = {
    "jaw": slice(0, 17),
    "eyebrows": slice(17, 27),
    "nose": slice(27, 36),
    "eyes": slice(36, 48),
    "mouth": slice(48, 68),
}


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
    if running_mode == "IMAGE":
        mode = vision.RunningMode.IMAGE
    elif running_mode == "VIDEO":
        mode = vision.RunningMode.VIDEO
    else:
        raise ValueError("running_mode must be IMAGE or VIDEO")

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


def result_to_keypoints(result):
    """Return eye-aligned 68 x/y/z keypoints, or None when no face is present."""
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    if len(landmarks) <= int(FEATURE_LANDMARK_INDICES.max()):
        return None

    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
    )
    if not np.isfinite(points).all():
        return None

    left_eye = points[LEFT_EYE_INDICES].mean(axis=0)
    right_eye = points[RIGHT_EYE_INDICES].mean(axis=0)
    eye_vector = right_eye[:2] - left_eye[:2]
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance <= 1e-6:
        return None

    selected = points[FEATURE_LANDMARK_INDICES].copy()
    selected -= (left_eye + right_eye) / 2.0
    selected /= eye_distance

    angle = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    x = selected[:, 0].copy()
    y = selected[:, 1].copy()
    selected[:, 0] = cosine * x + sine * y
    selected[:, 1] = -sine * x + cosine * y
    return selected.astype(np.float32)


def result_to_feature(result):
    keypoints = result_to_keypoints(result)
    return None if keypoints is None else keypoints.reshape(-1)


def is_plausible_face(image, result):
    """Permissive rejection of blank images and failed landmark geometry."""
    if image is None or not result.face_landmarks:
        return False, "no_face"
    if float(image.std()) < 8.0:
        return False, "low_contrast"

    points = np.asarray(
        [[point.x, point.y, point.z] for point in result.face_landmarks[0]],
        dtype=np.float32,
    )
    if not np.isfinite(points).all() or len(points) <= int(RIGHT_EYE_INDICES.max()):
        return False, "invalid_landmarks"

    xy = points[:, :2]
    span = xy.max(axis=0) - xy.min(axis=0)
    center = xy.mean(axis=0)
    left_eye = xy[LEFT_EYE_INDICES].mean(axis=0)
    right_eye = xy[RIGHT_EYE_INDICES].mean(axis=0)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    mouth_center = xy[[61, 291, 13, 14]].mean(axis=0)
    plausible = (
        0.15 <= span[0] <= 1.20
        and 0.18 <= span[1] <= 1.20
        and 0.02 <= center[0] <= 0.98
        and 0.02 <= center[1] <= 0.98
        and 0.06 <= eye_distance <= 0.65
        and left_eye[0] < right_eye[0]
        and mouth_center[1] > (left_eye[1] + right_eye[1]) / 2.0
    )
    return (True, "accepted") if plausible else (False, "implausible_geometry")


def landmarks_to_pixels(result, frame_shape):
    if not result.face_landmarks:
        return None
    height, width = frame_shape[:2]
    landmarks = result.face_landmarks[0]
    return [
        (int(landmarks[index].x * width), int(landmarks[index].y * height))
        for index in FEATURE_LANDMARK_INDICES
    ]
