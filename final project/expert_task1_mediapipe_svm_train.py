from pathlib import Path
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from expert_mediapipe_landmarks import (
    FEATURE_VERSION,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    create_face_landmarker,
    image_from_bgr,
    result_to_feature,
)


FORCE_REBUILD_FEATURES = False
# After automatic image/landmark filtering, downsample every class to the same
# valid-sample count. The SVM itself still uses class_weight=None.
BALANCE_TRAINING = True
BALANCE_RANDOM_SEED = 42
TARGET_IMAGE_SIZE = 224
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "facial_expression_dataset"
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
CACHE_PATH = BASE_DIR / "expert_task1_mediapipe_svm_features.npz"
MODEL_PATH = BASE_DIR / "expert_task1_mediapipe_svm.pkl"
MATRIX_PATH = BASE_DIR / "expert_task1_mediapipe_svm_confusion_matrix.png"
CACHE_VERSION = 2


def image_paths(split, label):
    class_dir = DATASET_DIR / split / label
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")
    return sorted(
        path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def is_plausible_face(image, result):
    """Reject blank, non-face, and visibly failed landmark detections."""
    if not result.face_landmarks:
        return False, "no_mediapipe_face"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 12.0:
        return False, "low_image_contrast"

    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in result.face_landmarks[0]],
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
        0.20 <= span[0] <= 0.95
        and 0.25 <= span[1] <= 0.95
        and 0.15 <= center[0] <= 0.85
        and 0.15 <= center[1] <= 0.85
        and 0.08 <= eye_distance <= 0.55
        and left_eye[0] < right_eye[0]
        and mouth_center[1] > (left_eye[1] + right_eye[1]) / 2.0
    )
    return (True, "accepted") if plausible else (False, "implausible_face_geometry")


def extract_feature(image_path, landmarker):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "read_failed"
    image = cv2.resize(image, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
    result = landmarker.detect(image_from_bgr(image))
    accepted, reason = is_plausible_face(image, result)
    if not accepted:
        return None, reason
    feature = result_to_feature(result)
    return (feature, "accepted") if feature is not None else (None, "feature_failed")


def build_split(split, landmarker):
    features_by_class = []
    for label in LABELS:
        paths = image_paths(split, label)
        features = []
        stats = {}
        for index, path in enumerate(paths, start=1):
            feature, status = extract_feature(path, landmarker)
            stats[status] = stats.get(status, 0) + 1
            if feature is not None:
                features.append(feature)
            if index % 500 == 0:
                print(f"[{split}] {label}: {index}/{len(paths)}")
        print(f"[{split}] {label}: {len(features)}/{len(paths)} accepted | {stats}")
        features_by_class.append(features)

    all_features, all_labels = [], []
    if split == "train" and BALANCE_TRAINING:
        limit = min(len(features) for features in features_by_class)
        rng = np.random.default_rng(BALANCE_RANDOM_SEED)
        print(f"[train] balancing each class to {limit} detected samples")
        for label_index, features in enumerate(features_by_class):
            for feature_index in rng.choice(len(features), size=limit, replace=False):
                all_features.append(features[feature_index])
                all_labels.append(label_index)
    else:
        for label_index, features in enumerate(features_by_class):
            all_features.extend(features)
            all_labels.extend([label_index] * len(features))

    X = np.vstack(all_features).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int32)
    print(f"[{split}] counts: {dict(zip(LABELS, np.bincount(y, minlength=len(LABELS)).tolist()))}")
    return X, y


def load_or_extract_features():
    if CACHE_PATH.exists() and not FORCE_REBUILD_FEATURES:
        data = np.load(CACHE_PATH)
        if int(data["cache_version"]) == CACHE_VERSION:
            print(f"Loading cache: {CACHE_PATH}")
            return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
    with create_face_landmarker(LANDMARKER_MODEL_PATH, "IMAGE") as landmarker:
        X_train, y_train = build_split("train", landmarker)
        X_test, y_test = build_split("test", landmarker)
    np.savez_compressed(
        CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cache_version=np.asarray(CACHE_VERSION, dtype=np.int32),
    )
    print(f"Saved feature cache: {CACHE_PATH}")
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_or_extract_features()
    print(f"X_train={X_train.shape}, X_test={X_test.shape}")
    model = Pipeline(
        [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight=None))]
    )
    print("Training unweighted RBF SVM ...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=LABELS, digits=4))
    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(matrix)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    figure, axis = plt.subplots(figsize=(9, 8))
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(MATRIX_PATH, dpi=160)
    plt.close(figure)
    start = time.perf_counter()
    model.predict(X_test[: min(1000, len(X_test))])
    print(f"SVM prediction time: {(time.perf_counter() - start) / min(1000, len(X_test)) * 1000:.4f} ms/image")
    with open(MODEL_PATH, "wb") as output:
        pickle.dump({"model": model, "labels": LABELS, "feature_type": FEATURE_VERSION}, output)
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
