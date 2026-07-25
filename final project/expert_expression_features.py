"""Shared landmark features for Expert Task 1 training and Task 2 inference."""

import numpy as np


FEATURE_VERSION = "eye_aligned_68_landmarks_plus_geometry_v1"


def _distance(points, first, second):
    return float(np.linalg.norm(points[first] - points[second]))


def _eye_aspect_ratio(points, outer, upper_a, upper_b, inner, lower_b, lower_a):
    width = _distance(points, outer, inner)
    if width <= 1e-6:
        return 0.0
    height = (_distance(points, upper_a, lower_a) + _distance(points, upper_b, lower_b)) / 2.0
    return height / width


def build_expression_feature(points):
    """Build a rotation-, translation-, and scale-normalized keypoint feature.

    Input is only the 68 LBF landmarks. No image pixels or expression-specific
    correction rules are used here.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return None

    left_eye_center = points[36:42].mean(axis=0)
    right_eye_center = points[42:48].mean(axis=0)
    eye_vector = right_eye_center - left_eye_center
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance <= 1e-6:
        return None

    # Rotate the eye line to horizontal, then normalize all distances by it.
    cosine = eye_vector[0] / eye_distance
    sine = eye_vector[1] / eye_distance
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    aligned = ((points - (left_eye_center + right_eye_center) / 2.0) @ rotation.T) / eye_distance

    left_eye_ratio = _eye_aspect_ratio(aligned, 36, 37, 38, 39, 40, 41)
    right_eye_ratio = _eye_aspect_ratio(aligned, 42, 43, 44, 45, 46, 47)
    mouth_width = _distance(aligned, 48, 54)
    outer_mouth_opening = _distance(aligned, 51, 57)
    inner_mouth_opening = _distance(aligned, 62, 66)
    if mouth_width <= 1e-6:
        return None

    mouth_center_y = float((aligned[51, 1] + aligned[57, 1]) / 2.0)
    corner_center_y = float((aligned[48, 1] + aligned[54, 1]) / 2.0)
    corner_lift = (mouth_center_y - corner_center_y) / mouth_width
    left_brow_eye_gap = float(aligned[36:42, 1].mean() - aligned[17:22, 1].mean())
    right_brow_eye_gap = float(aligned[42:48, 1].mean() - aligned[22:27, 1].mean())

    geometry = np.array(
        [
            left_eye_ratio,
            right_eye_ratio,
            mouth_width,
            outer_mouth_opening / mouth_width,
            inner_mouth_opening / mouth_width,
            corner_lift,
            left_brow_eye_gap,
            right_brow_eye_gap,
            _distance(aligned, 21, 22),
            float(aligned[21, 1] - aligned[17, 1]),
            float(aligned[26, 1] - aligned[22, 1]),
            float(aligned[33, 1] - aligned[51, 1]),
        ],
        dtype=np.float32,
    )
    return np.concatenate((aligned.flatten(), geometry)).astype(np.float32)
