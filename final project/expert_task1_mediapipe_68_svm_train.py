"""Expert Task 1: balanced MediaPipe 68-point expression classification by SVM."""

from pathlib import Path
import pickle
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from expert_mediapipe_68_svm import (
    FEATURE_DIMENSION,
    FEATURE_VERSION,
    create_face_landmarker,
    image_from_bgr,
    is_plausible_face,
    result_to_feature,
)


RANDOM_SEED = 42
TARGET_IMAGE_SIZE = 224
VALIDATION_FRACTION = 0.20
FORCE_REBUILD_FEATURES = False
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "facial_expression_dataset"
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
CACHE_PATH = BASE_DIR / "expert_task1_mediapipe_68_svm_features.npz"
MODEL_PATH = BASE_DIR / "expert_task1_mediapipe_68_expression_svm.pkl"
MATRIX_PATH = BASE_DIR / "expert_task1_mediapipe_68_svm_confusion_matrix.png"
CACHE_VERSION = 1


def image_paths(split, label):
    class_dir = DATASET_DIR / split / label
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def extract_feature(image_path, landmarker):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "read_failed"
    image = cv2.resize(
        image, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
    )
    result = landmarker.detect(image_from_bgr(image))
    accepted, reason = is_plausible_face(image, result)
    if not accepted:
        return None, reason
    feature = result_to_feature(result)
    return (feature, "accepted") if feature is not None else (None, "feature_failed")


def build_split(split, landmarker):
    features = []
    targets = []
    for label_index, label in enumerate(LABELS):
        paths = image_paths(split, label)
        statistics = {}
        accepted = 0
        for index, path in enumerate(paths, start=1):
            feature, status = extract_feature(path, landmarker)
            statistics[status] = statistics.get(status, 0) + 1
            if feature is not None:
                features.append(feature)
                targets.append(label_index)
                accepted += 1
            if index % 500 == 0:
                print(f"[{split}] {label}: {index}/{len(paths)}")
        print(f"[{split}] {label}: {accepted}/{len(paths)} accepted | {statistics}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int32)
    if X.ndim != 2 or X.shape[1] != FEATURE_DIMENSION:
        raise RuntimeError(f"Unexpected feature shape: {X.shape}")
    counts = np.bincount(y, minlength=len(LABELS))
    if np.any(counts == 0):
        raise RuntimeError(f"At least one class has no valid faces: {counts.tolist()}")
    print(f"[{split}] accepted counts: {dict(zip(LABELS, counts.tolist()))}")
    return X, y


def load_or_extract_features():
    if CACHE_PATH.exists() and not FORCE_REBUILD_FEATURES:
        data = np.load(CACHE_PATH)
        compatible = (
            int(data["cache_version"]) == CACHE_VERSION
            and str(data["feature_version"]) == FEATURE_VERSION
        )
        if compatible:
            print(f"Loading cached SVM keypoints: {CACHE_PATH}")
            return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
        print("Existing SVM cache is incompatible and will be rebuilt.")

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
        feature_version=np.asarray(FEATURE_VERSION),
    )
    print(f"Saved SVM feature cache: {CACHE_PATH}")
    return X_train, y_train, X_test, y_test


def make_balanced_training_set(X, y):
    rng = np.random.default_rng(RANDOM_SEED)
    indices_by_class = [
        np.flatnonzero(y == class_index) for class_index in range(len(LABELS))
    ]
    samples_per_class = min(len(indices) for indices in indices_by_class)
    selected = np.concatenate(
        [
            rng.choice(indices, size=samples_per_class, replace=False)
            for indices in indices_by_class
        ]
    )
    rng.shuffle(selected)
    balanced_X = X[selected]
    balanced_y = y[selected]
    counts = np.bincount(balanced_y, minlength=len(LABELS))
    print(f"Balanced training counts: {dict(zip(LABELS, counts.tolist()))}")
    return balanced_X, balanced_y


def make_pipeline(C, gamma):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight=None,
                    cache_size=2048,
                ),
            ),
        ]
    )


def select_hyperparameters(X_train, y_train, X_validation, y_validation):
    candidates_C = [1.0, 3.0, 10.0, 30.0, 100.0]
    candidates_gamma = ["scale", 0.001, 0.003, 0.01, 0.03]
    best_parameters = None
    best_score = -1.0
    for C in candidates_C:
        for gamma in candidates_gamma:
            candidate = make_pipeline(C, gamma)
            candidate.fit(X_train, y_train)
            predictions = candidate.predict(X_validation)
            macro_f1 = f1_score(
                y_validation, predictions, average="macro", zero_division=0
            )
            accuracy = accuracy_score(y_validation, predictions)
            print(
                f"C={C:<5g} gamma={str(gamma):<5} | "
                f"val_accuracy={accuracy:.4f} val_macro_f1={macro_f1:.4f}"
            )
            if macro_f1 > best_score:
                best_score = macro_f1
                best_parameters = (C, gamma)
    print(
        f"Best validation parameters: C={best_parameters[0]}, "
        f"gamma={best_parameters[1]}, macro_f1={best_score:.4f}"
    )
    return best_parameters, best_score


def measure_single_prediction_latency(model, sample, warmup=100, runs=1000):
    """Measure StandardScaler + SVM batch=1 prediction latency."""
    sample = sample.reshape(1, -1)
    for _ in range(warmup):
        model.predict(sample)
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(sample)
    return (time.perf_counter() - start) * 1000.0 / runs


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print("Classifier: StandardScaler + RBF SVM")
    print("Input: 68 x/y/z MediaPipe keypoints (204 values); no image pixels.")
    print("No class weights, output correction, or probability adjustment.")

    X_train_all, y_train_all, X_test, y_test = load_or_extract_features()
    X_balanced, y_balanced = make_balanced_training_set(X_train_all, y_train_all)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_balanced,
        y_balanced,
        test_size=VALIDATION_FRACTION,
        stratify=y_balanced,
        random_state=RANDOM_SEED,
    )
    best_parameters, best_validation_f1 = select_hyperparameters(
        X_train, y_train, X_validation, y_validation
    )

    print("Fitting final SVM on all balanced training samples ...")
    model = make_pipeline(*best_parameters)
    model.fit(X_balanced, y_balanced)

    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    test_macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    single_prediction_ms = measure_single_prediction_latency(model, X_test[0])

    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=LABELS, digits=4))
    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(matrix)
    print(
        "Batch=1 SVM prediction latency: "
        f"{single_prediction_ms:.4f} ms/prediction"
    )

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    figure, axis = plt.subplots(figsize=(9, 8))
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(MATRIX_PATH, dpi=160)
    plt.close(figure)

    neutral_reference = X_train_all[y_train_all == LABELS.index("neutral")].mean(
        axis=0
    )
    payload = {
        "classifier": "RBF SVM",
        "model": model,
        "labels": LABELS,
        "feature_version": FEATURE_VERSION,
        "balanced_samples_per_class": int(np.bincount(y_balanced).min()),
        "best_C": best_parameters[0],
        "best_gamma": best_parameters[1],
        "best_validation_macro_f1": best_validation_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "single_prediction_ms": single_prediction_ms,
        "neutral_reference": neutral_reference.astype(np.float32),
    }
    with open(MODEL_PATH, "wb") as output:
        pickle.dump(payload, output)
    print(f"Saved SVM model: {MODEL_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")


if __name__ == "__main__":
    main()
