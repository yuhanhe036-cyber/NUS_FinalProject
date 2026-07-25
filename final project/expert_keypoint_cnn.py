"""MediaPipe keypoint extraction and 1D CNN shared by Expert Tasks 1 and 2."""

from pathlib import Path
import shutil
from urllib.request import urlopen

import mediapipe as mp
import numpy as np
import torch
from torch import nn


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# MediaPipe points mapped to the conventional 68-point landmark anatomy:
# jaw 17, eyebrows 10, nose 9, eyes 12 and mouth 20.
FEATURE_LANDMARK_INDICES = np.asarray(
    [
        # Jaw: left cheek to chin to right cheek (17)
        234, 93, 132, 58, 172, 136, 150, 176, 152,
        400, 378, 379, 365, 397, 288, 361, 454,
        # Eyebrows (10)
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        # Nose: bridge followed by nostril contour (9)
        168, 6, 197, 195, 98, 97, 2, 326, 327,
        # Eyes (12)
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        # Mouth: outer contour 12 followed by inner contour 8 (20)
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
REGION_CONV_LAYERS = 5
LANDMARK_REGION_SLICES = {
    "jaw": slice(0, 17),
    "eyebrows": slice(17, 27),
    "nose": slice(27, 36),
    "eyes": slice(36, 48),
    "mouth": slice(48, 68),
}


def ensure_face_landmarker_model(model_path):
    """Download the official MediaPipe model once when it is not present."""
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
    """Return 68 aligned 3D keypoints without using image-pixel features."""
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
    eye_midpoint = (left_eye + right_eye) / 2.0
    selected -= eye_midpoint
    selected /= eye_distance

    # Rotate x/y so head tilt does not become an expression cue.
    angle = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    x = selected[:, 0].copy()
    y = selected[:, 1].copy()
    selected[:, 0] = cosine * x + sine * y
    selected[:, 1] = -sine * x + cosine * y
    return selected.astype(np.float32)


def is_plausible_face(image, result):
    """Reject unreadable/non-face detections while keeping the filter permissive."""
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


def landmarks_to_pixels(result, frame_shape, selected_only=True):
    if not result.face_landmarks:
        return None
    height, width = frame_shape[:2]
    landmarks = result.face_landmarks[0]
    indices = FEATURE_LANDMARK_INDICES if selected_only else range(len(landmarks))
    return [
        (int(landmarks[index].x * width), int(landmarks[index].y * height))
        for index in indices
    ]


def standardize_keypoints(keypoints, feature_mean, feature_std):
    flat = np.asarray(keypoints, dtype=np.float32).reshape(-1)
    standardized = (flat - np.asarray(feature_mean, dtype=np.float32)) / np.asarray(
        feature_std, dtype=np.float32
    )
    # Conv1d input is channels x sequence: x/y/z x 68 keypoints.
    return torch.from_numpy(standardized.reshape(NUM_KEYPOINTS, 3).T.copy())


class RegionEncoder(nn.Module):
    """Encode one facial region into a fixed-size representation."""

    def __init__(self, embedding_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.08),
            nn.Conv1d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.12),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(96 * 4, embedding_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.network(inputs)


class KeypointExpressionCNN(nn.Module):
    """Five-branch CNN with five Conv1d layers in every facial region."""

    def __init__(self, num_classes):
        super().__init__()
        embedding_size = 64
        self.region_encoders = nn.ModuleDict(
            {
                region_name: RegionEncoder(embedding_size)
                for region_name in LANDMARK_REGION_SLICES
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_size * len(LANDMARK_REGION_SLICES)),
            nn.Dropout(0.30),
            nn.Linear(embedding_size * len(LANDMARK_REGION_SLICES), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, keypoints):
        region_features = [
            self.region_encoders[region_name](keypoints[:, :, region_slice])
            for region_name, region_slice in LANDMARK_REGION_SLICES.items()
        ]
        return self.classifier(torch.cat(region_features, dim=1))
