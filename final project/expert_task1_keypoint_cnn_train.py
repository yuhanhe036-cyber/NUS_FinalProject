"""Expert Task 1: MediaPipe facial keypoints classified by a 1D CNN."""

from copy import deepcopy
from pathlib import Path
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from expert_keypoint_cnn import (
    FEATURE_DIMENSION,
    FEATURE_VERSION,
    KeypointExpressionCNN,
    LANDMARK_REGION_SLICES,
    NUM_KEYPOINTS,
    REGION_CONV_LAYERS,
    create_face_landmarker,
    image_from_bgr,
    is_plausible_face,
    result_to_keypoints,
    standardize_keypoints,
)


RANDOM_SEED = 42
TARGET_IMAGE_SIZE = 224
VALIDATION_FRACTION = 0.15
SAMPLES_PER_CLASS_PER_BATCH = 16
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0
FORCE_REBUILD_FEATURES = False
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "facial_expression_dataset"
LANDMARKER_MODEL_PATH = BASE_DIR / "mediapipe_face_landmarker.task"
CACHE_PATH = BASE_DIR / "expert_task1_keypoint_cnn_features.npz"
MODEL_PATH = BASE_DIR / "expert_task1_keypoint_cnn_5conv.pt"
MATRIX_PATH = BASE_DIR / "expert_task1_keypoint_cnn_5conv_confusion_matrix.png"
HISTORY_PATH = BASE_DIR / "expert_task1_keypoint_cnn_5conv_history.png"
CACHE_VERSION = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_paths(split, label):
    class_dir = DATASET_DIR / split / label
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def extract_keypoint(image_path, landmarker):
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
    keypoints = result_to_keypoints(result)
    if keypoints is None:
        return None, "normalization_failed"
    return keypoints.reshape(-1), "accepted"


def build_split(split, landmarker):
    all_features = []
    all_labels = []
    for label_index, label in enumerate(LABELS):
        paths = image_paths(split, label)
        statistics = {}
        accepted = 0
        for index, path in enumerate(paths, start=1):
            feature, status = extract_keypoint(path, landmarker)
            statistics[status] = statistics.get(status, 0) + 1
            if feature is not None:
                all_features.append(feature)
                all_labels.append(label_index)
                accepted += 1
            if index % 500 == 0:
                print(f"[{split}] {label}: {index}/{len(paths)}")
        print(f"[{split}] {label}: {accepted}/{len(paths)} accepted | {statistics}")

    features = np.asarray(all_features, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise RuntimeError(f"Unexpected extracted feature shape: {features.shape}")
    counts = np.bincount(labels, minlength=len(LABELS))
    if np.any(counts == 0):
        raise RuntimeError(f"At least one class has no valid faces: {counts.tolist()}")
    print(f"[{split}] accepted counts: {dict(zip(LABELS, counts.tolist()))}")
    return features, labels


def load_or_extract_features():
    if CACHE_PATH.exists() and not FORCE_REBUILD_FEATURES:
        data = np.load(CACHE_PATH)
        compatible = (
            int(data["cache_version"]) == CACHE_VERSION
            and str(data["feature_version"]) == FEATURE_VERSION
        )
        if compatible:
            print(f"Loading cached keypoints: {CACHE_PATH}")
            return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
        print("Existing cache is incompatible and will be rebuilt.")

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
    print(f"Saved keypoint-only cache: {CACHE_PATH}")
    return X_train, y_train, X_test, y_test


class KeypointDataset(Dataset):
    def __init__(self, features, labels, feature_mean, feature_std, augment=False):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        keypoints = self.features[index].reshape(NUM_KEYPOINTS, 3).copy()
        if self.augment:
            # Small geometric noise improves webcam robustness without changing labels.
            angle = np.random.uniform(-0.04, 0.04)
            cosine, sine = np.cos(angle), np.sin(angle)
            x = keypoints[:, 0].copy()
            y = keypoints[:, 1].copy()
            keypoints[:, 0] = cosine * x - sine * y
            keypoints[:, 1] = sine * x + cosine * y
            keypoints += np.random.normal(0.0, 0.004, keypoints.shape).astype(np.float32)
        tensor = standardize_keypoints(keypoints, self.feature_mean, self.feature_std)
        return tensor, int(self.labels[index])


class BalancedClassBatchSampler(Sampler):
    """Put exactly the same number of every expression in each training batch."""

    def __init__(self, labels, samples_per_class, seed):
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        self.indices_by_class = [
            np.flatnonzero(labels == class_index) for class_index in range(len(LABELS))
        ]
        if any(len(indices) == 0 for indices in self.indices_by_class):
            raise ValueError("Every class must contain at least one training sample.")
        largest = max(len(indices) for indices in self.indices_by_class)
        self.steps_per_epoch = int(np.ceil(largest / samples_per_class))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps_per_epoch):
            batch = []
            for indices in self.indices_by_class:
                selected = rng.choice(
                    indices,
                    size=self.samples_per_class,
                    replace=len(indices) < self.samples_per_class,
                )
                batch.extend(selected.tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.steps_per_epoch


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for keypoints, labels in loader:
            logits = model(keypoints.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    accuracy = float(np.mean(targets == predictions))
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return accuracy, macro_f1, targets, predictions


def measure_single_prediction_latency(model, sample, device, warmup=100, runs=1000):
    """Measure batch=1 latency until the predicted class is available on CPU."""
    model.eval()
    sample = sample.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample).argmax(dim=1).item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(sample).argmax(dim=1).item()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs


def save_history(losses, validation_scores):
    epochs = range(1, len(losses) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, color="tab:red")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, validation_scores, color="tab:blue")
    axes[1].set(title="Validation Macro F1", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(HISTORY_PATH, dpi=160)
    plt.close(figure)


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Input: {NUM_KEYPOINTS} x/y/z MediaPipe keypoints in the conventional "
        "68-point anatomy layout. No face-image pixels enter the CNN."
    )
    print(
        f"Architecture: {len(LANDMARK_REGION_SLICES)} parallel facial-region "
        "branches, "
        f"{REGION_CONV_LAYERS} Conv1d layers per branch."
    )

    X_train_all, y_train_all, X_test, y_test = load_or_extract_features()
    all_indices = np.arange(len(y_train_all))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_FRACTION,
        stratify=y_train_all,
        random_state=RANDOM_SEED,
    )

    feature_mean = X_train_all[train_indices].mean(axis=0).astype(np.float32)
    feature_std = X_train_all[train_indices].std(axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-5)
    neutral_index = LABELS.index("neutral")
    neutral_reference = X_train_all[train_indices][
        y_train_all[train_indices] == neutral_index
    ].mean(axis=0).astype(np.float32)

    training_dataset = KeypointDataset(
        X_train_all[train_indices],
        y_train_all[train_indices],
        feature_mean,
        feature_std,
        augment=True,
    )
    validation_dataset = KeypointDataset(
        X_train_all[validation_indices],
        y_train_all[validation_indices],
        feature_mean,
        feature_std,
    )
    test_dataset = KeypointDataset(X_test, y_test, feature_mean, feature_std)
    sampler = BalancedClassBatchSampler(
        y_train_all[train_indices], SAMPLES_PER_CLASS_PER_BATCH, RANDOM_SEED
    )
    print(
        "Each optimization batch: "
        f"{dict.fromkeys(LABELS, SAMPLES_PER_CLASS_PER_BATCH)}"
    )
    print("Loss: ordinary CrossEntropyLoss with no class weights.")

    train_loader = DataLoader(
        training_dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = KeypointExpressionCNN(len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_state = None
    best_validation_f1 = -1.0
    epochs_without_improvement = 0
    losses = []
    validation_scores = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for keypoints, labels in train_loader:
            keypoints = keypoints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(keypoints), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        validation_accuracy, validation_f1, _, _ = evaluate(
            model, validation_loader, device
        )
        scheduler.step(validation_f1)
        losses.append(average_loss)
        validation_scores.append(validation_f1)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={average_loss:.4f} | "
            f"val_accuracy={validation_accuracy:.4f} | val_macro_f1={validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation macro F1 stopped improving.")
                break

    model.load_state_dict(best_state)
    save_history(losses, validation_scores)
    test_accuracy, test_macro_f1, y_true, y_pred = evaluate(model, test_loader, device)
    single_prediction_ms = measure_single_prediction_latency(
        model, test_dataset[0][0], device
    )

    print(f"\nBest validation macro F1: {best_validation_f1:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))
    matrix = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(matrix)
    print(
        "Batch=1 CNN prediction latency: "
        f"{single_prediction_ms:.4f} ms/prediction "
        "(includes argmax and synchronized result)"
    )

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    figure, axis = plt.subplots(figsize=(9, 8))
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(MATRIX_PATH, dpi=160)
    plt.close(figure)

    torch.save(
        {
            "architecture": "FiveBranchKeypointExpressionCNN",
            "region_conv_layers": REGION_CONV_LAYERS,
            "state_dict": model.state_dict(),
            "labels": LABELS,
            "feature_version": FEATURE_VERSION,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "neutral_reference": neutral_reference.tolist(),
            "best_validation_macro_f1": best_validation_f1,
            "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "single_prediction_ms": single_prediction_ms,
        },
        MODEL_PATH,
    )
    print(f"Saved keypoint CNN model: {MODEL_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")
    print(f"Saved training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
