"""Real-time Apple emoji effects driven by the Expert keypoint CNN.

The CNN prediction is used directly. No class-specific probability weighting,
thresholding, or label correction is applied.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import torch

from expert_keypoint_cnn import (
    FEATURE_VERSION,
    KeypointExpressionCNN,
    REGION_CONV_LAYERS,
    create_face_landmarker,
    image_from_bgr,
    result_to_keypoints,
    standardize_keypoints,
)


BASE_DIR = Path(__file__).resolve().parent
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
MODEL_PATH = BASE_DIR / "expert_task1_keypoint_cnn_5conv.pt"
EMOJI_DIR = BASE_DIR / "apple_emoji_assets"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_NAME = "Expert Task 2 - 3D Emoji Effects"
POSE_SMOOTHING = 0.28
CALIBRATION_FRAMES = 36
POSE_DEAD_ZONE_DEGREES = 2.2

EMOJI_FILES = {
    "happy": "happy.png",
    "disgust": "disgust.png",
    "fear": "fear.png",
    "sad": "sad.png",
    "surprise": "surprise.png",
    "angry": "angry.png",
}

# MediaPipe indices used for a stable six-point head-pose estimate.
POSE_LANDMARK_INDICES = np.asarray([1, 152, 33, 263, 61, 291], dtype=np.int32)
HEAD_MODEL_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (-225.0, 170.0, -135.0),  # left eye outer corner
        (225.0, 170.0, -135.0),   # right eye outer corner
        (-150.0, -150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find CNN model: {MODEL_PATH}\n"
            "Run expert_task1_keypoint_cnn_train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model and keypoint extractor versions do not match.")
    if payload.get("region_conv_layers") != REGION_CONV_LAYERS:
        raise ValueError("The checkpoint is not the expected five-layer region CNN.")

    labels = payload["labels"]
    model = KeypointExpressionCNN(len(labels)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    if "neutral_reference" not in payload:
        raise ValueError(
            "The model has no neutral reference. Run the updated "
            "expert_task1_keypoint_cnn_train.py first."
        )
    neutral_reference = np.asarray(
        payload["neutral_reference"], dtype=np.float32
    ).reshape(-1, 3)
    return model, labels, feature_mean, feature_std, neutral_reference, device


def load_emoji_assets():
    assets = {}
    missing = []
    for label, filename in EMOJI_FILES.items():
        path = EMOJI_DIR / filename
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            missing.append(str(path))
            continue
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported emoji image format: {path}")
        if image.shape[2] == 3:
            alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=2)
        assets[label] = image

    if missing:
        raise FileNotFoundError(
            "Missing Apple emoji assets:\n" + "\n".join(missing)
        )
    return assets


def rotation_matrix_xyz(pitch, yaw, roll):
    pitch, yaw, roll = np.radians([pitch, yaw, roll])
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rotate_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    rotate_y = np.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rotate_z = np.asarray(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rotate_z @ rotate_y @ rotate_x


def perspective_rotate_emoji(image, size, pitch, yaw, roll):
    """Warp one RGBA emoji as a flat 3D object under the current head pose."""
    source_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    source_corners = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    plane_corners = np.asarray(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        dtype=np.float32,
    )

    rotated = plane_corners @ rotation_matrix_xyz(pitch, yaw, roll).T
    camera_distance = 4.0
    depth = np.maximum(camera_distance - rotated[:, 2], 0.5)
    projected = rotated[:, :2] / depth[:, None]

    canvas_size = int(np.ceil(size * 1.55))
    projected *= camera_distance * size * 0.47
    projected += canvas_size / 2.0
    transform = cv2.getPerspectiveTransform(source_corners, projected.astype(np.float32))
    return cv2.warpPerspective(
        source_image,
        transform,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def alpha_blend_rgba(frame, overlay, center, opacity=1.0):
    overlay_height, overlay_width = overlay.shape[:2]
    left = int(center[0] - overlay_width / 2)
    top = int(center[1] - overlay_height / 2)
    right = left + overlay_width
    bottom = top + overlay_height

    frame_height, frame_width = frame.shape[:2]
    clipped_left = max(left, 0)
    clipped_top = max(top, 0)
    clipped_right = min(right, frame_width)
    clipped_bottom = min(bottom, frame_height)
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return

    overlay_x1 = clipped_left - left
    overlay_y1 = clipped_top - top
    overlay_x2 = overlay_x1 + clipped_right - clipped_left
    overlay_y2 = overlay_y1 + clipped_bottom - clipped_top
    cropped = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    alpha = cropped[:, :, 3:4].astype(np.float32) / 255.0
    alpha *= opacity

    region = frame[clipped_top:clipped_bottom, clipped_left:clipped_right]
    blended = cropped[:, :, :3].astype(np.float32) * alpha
    blended += region.astype(np.float32) * (1.0 - alpha)
    region[:] = np.clip(blended, 0, 255).astype(np.uint8)


def face_exclusion_ellipse(result, frame_shape, mirrored=True):
    if not result.face_landmarks:
        return None

    height, width = frame_shape[:2]
    points = np.asarray(
        [[landmark.x * width, landmark.y * height]
         for landmark in result.face_landmarks[0]],
        dtype=np.float32,
    )
    if mirrored:
        points[:, 0] = width - 1 - points[:, 0]

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radii = np.maximum((maximum - minimum) * np.asarray([0.78, 0.82]), 1.0)
    return center, radii


def is_inside_face_ellipse(point, ellipse):
    if ellipse is None:
        return False
    center, radii = ellipse
    normalized = (np.asarray(point, dtype=np.float32) - center) / radii
    return float(np.dot(normalized, normalized)) <= 1.0


def render_emoji_background(frame, emoji, pose, face_ellipse, timestamp):
    pitch, yaw, roll = pose
    small = perspective_rotate_emoji(emoji, 48, pitch, yaw, roll)
    large = perspective_rotate_emoji(emoji, 62, pitch, yaw, roll)

    height, width = frame.shape[:2]
    spacing_x = 92
    spacing_y = 86
    phase_x = int((timestamp * 8.0) % spacing_x)
    phase_y = int((timestamp * 5.0) % spacing_y)

    row = 0
    for y in range(-spacing_y + phase_y, height + spacing_y, spacing_y):
        row_offset = spacing_x // 2 if row % 2 else 0
        column = 0
        for x in range(
            -spacing_x + phase_x + row_offset,
            width + spacing_x,
            spacing_x,
        ):
            center = (x, y)
            if not is_inside_face_ellipse(center, face_ellipse):
                sprite = large if (row + column) % 4 == 0 else small
                alpha_blend_rgba(frame, sprite, center, opacity=0.92)
            column += 1
        row += 1


class HeadPoseTracker:
    def __init__(self, smoothing=POSE_SMOOTHING):
        self.smoothing = smoothing
        self.pose = None
        self.rotation_vector = None
        self.translation_vector = None

    def update(self, result, frame_shape):
        if not result.face_landmarks:
            self.reset()
            return None

        landmarks = result.face_landmarks[0]
        if len(landmarks) <= int(POSE_LANDMARK_INDICES.max()):
            self.reset()
            return None

        height, width = frame_shape[:2]
        image_points = np.asarray(
            [
                (landmarks[index].x * width, landmarks[index].y * height)
                for index in POSE_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )
        focal_length = float(width)
        camera_matrix = np.asarray(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)

        use_guess = self.rotation_vector is not None
        success, rotation_vector, translation_vector = cv2.solvePnP(
            HEAD_MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            self.rotation_vector,
            self.translation_vector,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self.reset()
            return None

        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, _ = cv2.RQDecomp3x3(rotation_matrix)[0]
        eye_vector = image_points[3] - image_points[2]
        roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

        measured = np.asarray(
            [
                np.clip(pitch, -35.0, 35.0),
                np.clip(yaw, -50.0, 50.0),
                np.clip(roll, -40.0, 40.0),
            ],
            dtype=np.float32,
        )
        if self.pose is None:
            self.pose = measured
        else:
            self.pose = (
                self.smoothing * measured + (1.0 - self.smoothing) * self.pose
            )
        return tuple(float(value) for value in self.pose)

    def reset(self):
        self.pose = None
        self.rotation_vector = None
        self.translation_vector = None


class PersonalCalibration:
    """Align neutral geometry and the pose origin to the current user."""

    def __init__(self, neutral_reference, required_frames=CALIBRATION_FRAMES):
        self.neutral_reference = np.asarray(
            neutral_reference, dtype=np.float32
        ).reshape(-1, 3)
        self.required_frames = required_frames
        self.reset()

    @property
    def complete(self):
        return self.neutral_offset is not None and self.pose_origin is not None

    def add_sample(self, keypoints, pose):
        if self.complete or keypoints is None or pose is None:
            return self.complete

        self.keypoint_samples.append(np.asarray(keypoints, dtype=np.float32).copy())
        self.pose_samples.append(np.asarray(pose, dtype=np.float32))
        if len(self.keypoint_samples) < self.required_frames:
            return False

        personal_neutral = np.median(
            np.stack(self.keypoint_samples), axis=0
        ).astype(np.float32)
        self.neutral_offset = self.neutral_reference - personal_neutral
        self.pose_origin = np.median(
            np.stack(self.pose_samples), axis=0
        ).astype(np.float32)
        print("Neutral expression and straight-head calibration completed.")
        return True

    def calibrate_keypoints(self, keypoints):
        if not self.complete:
            return None
        return np.asarray(keypoints, dtype=np.float32) + self.neutral_offset

    def calibrate_pose(self, pose):
        if not self.complete:
            return None
        corrected = np.asarray(pose, dtype=np.float32) - self.pose_origin
        magnitude = np.maximum(
            np.abs(corrected) - POSE_DEAD_ZONE_DEGREES, 0.0
        )
        corrected = np.sign(corrected) * magnitude
        corrected[0] = np.clip(corrected[0], -32.0, 32.0)
        corrected[1] = np.clip(corrected[1], -45.0, 45.0)
        corrected[2] = np.clip(corrected[2], -35.0, 35.0)
        return tuple(float(value) for value in corrected)

    def reset(self):
        self.keypoint_samples = []
        self.pose_samples = []
        self.neutral_offset = None
        self.pose_origin = None


def predict_expression(model, keypoints, feature_mean, feature_std, device, labels):
    tensor = standardize_keypoints(
        keypoints, feature_mean, feature_std
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        predicted_index = int(model(tensor).argmax(dim=1).item())
    return labels[predicted_index]


def main():
    cv2.setUseOptimized(True)
    (
        model,
        labels,
        feature_mean,
        feature_std,
        neutral_reference,
        device,
    ) = load_model()
    emoji_assets = load_emoji_assets()

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {CAMERA_INDEX}")

    pose_tracker = HeadPoseTracker()
    calibration = PersonalCalibration(neutral_reference)
    print(f"Using device: {device}. Press q or Esc to quit.")
    print(
        f"Look straight at the camera with a neutral expression for "
        f"{CALIBRATION_FRAMES} valid frames. Press c to recalibrate."
    )

    try:
        with create_face_landmarker(LANDMARKER_MODEL_PATH, "VIDEO") as landmarker:
            while True:
                ok, camera_frame = camera.read()
                if not ok:
                    break

                timestamp_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(
                    image_from_bgr(camera_frame), timestamp_ms
                )
                keypoints = result_to_keypoints(result)
                raw_pose = pose_tracker.update(result, camera_frame.shape)

                # Classification uses the original camera frame. Only the display
                # is mirrored, so the CNN receives the same landmark convention
                # used during training.
                display_frame = cv2.flip(camera_frame, 1)
                if not calibration.complete:
                    calibration.add_sample(keypoints, raw_pose)
                elif keypoints is not None and raw_pose is not None:
                    calibrated_keypoints = calibration.calibrate_keypoints(keypoints)
                    pose = calibration.calibrate_pose(raw_pose)
                    label = predict_expression(
                        model,
                        calibrated_keypoints,
                        feature_mean,
                        feature_std,
                        device,
                        labels,
                    )
                    if label != "neutral":
                        # Mirror yaw and roll to match the mirrored camera display.
                        display_pose = (pose[0], -pose[1], -pose[2])
                        face_ellipse = face_exclusion_ellipse(
                            result, display_frame.shape, mirrored=True
                        )
                        render_emoji_background(
                            display_frame,
                            emoji_assets[label],
                            display_pose,
                            face_ellipse,
                            time.monotonic(),
                        )

                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    calibration.reset()
                    pose_tracker.reset()
                    print(
                        "Recalibration started. Look straight at the camera "
                        "with a neutral expression."
                    )
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
