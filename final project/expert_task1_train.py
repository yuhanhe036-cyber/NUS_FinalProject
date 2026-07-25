from copy import deepcopy
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms

from expert_expression_cnn import (
    INPUT_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    ExpressionCNN,
)


# ===================== Settings =====================
RANDOM_SEED = 42
BATCH_SIZE = 128
SAMPLES_PER_CLASS_PER_BATCH = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Regularization only; no per-class loss weighting is used.
VALIDATION_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 0  # Required for reliable execution on Windows.

# The loss is ordinary CrossEntropyLoss: no class weights are supplied to it.
# Each optimization batch contains the same number of samples from every class.
# Minority examples are seen more often through fresh image augmentation, while
# source files remain untouched.
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# ====================================================


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "facial_expression_dataset"
MODEL_PATH = BASE_DIR / "expert_task1_expression_cnn.pt"
CONFUSION_MATRIX_PATH = BASE_DIR / "expert_task1_cnn_confusion_matrix.png"
HISTORY_PATH = BASE_DIR / "expert_task1_cnn_history.png"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FerImageDataset(Dataset):
    def __init__(self, split, transform):
        self.transform = transform
        self.samples = []
        for label_index, label in enumerate(LABELS):
            class_dir = DATASET_DIR / split / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            self.samples.extend((path, label_index) for path in image_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            return self.transform(image), label


def make_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.ToTensor(),
            transforms.Normalize((NORMALIZATION_MEAN,), (NORMALIZATION_STD,)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((NORMALIZATION_MEAN,), (NORMALIZATION_STD,)),
        ]
    )
    return train_transform, evaluation_transform


def class_counts(dataset):
    counts = np.zeros(len(LABELS), dtype=np.int32)
    for _, label in dataset.samples:
        counts[label] += 1
    return dict(zip(LABELS, counts.tolist()))


class BalancedClassBatchSampler(Sampler):
    """Create batches with equal class counts without changing the loss function."""

    def __init__(self, labels, samples_per_class, seed):
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        self.indices_by_class = [
            np.flatnonzero(labels == class_index) for class_index in range(len(LABELS))
        ]
        if any(len(indices) == 0 for indices in self.indices_by_class):
            raise ValueError("Every class needs at least one training sample.")
        largest_class_size = max(len(indices) for indices in self.indices_by_class)
        self.steps_per_epoch = int(np.ceil(largest_class_size / samples_per_class))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps_per_epoch):
            batch = []
            for indices in self.indices_by_class:
                sampled = rng.choice(
                    indices,
                    size=self.samples_per_class,
                    replace=len(indices) < self.samples_per_class,
                )
                batch.extend(sampled.tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.steps_per_epoch


def accuracy_and_predictions(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(targets)))
    return accuracy, np.asarray(targets), np.asarray(predictions)


def plot_history(train_losses, validation_macro_f1_scores):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, color="tab:red")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].plot(epochs, validation_macro_f1_scores, color="tab:blue")
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(HISTORY_PATH, dpi=160)
    plt.close(fig)


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Extract facial_expression_dataset.zip first."
        )

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, evaluation_transform = make_transforms()
    train_dataset_for_split = FerImageDataset("train", evaluation_transform)
    train_dataset_for_augmentation = FerImageDataset("train", train_transform)
    test_dataset = FerImageDataset("test", evaluation_transform)
    print(f"Raw train class counts: {class_counts(train_dataset_for_split)}")
    print(f"Raw test class counts:  {class_counts(test_dataset)}")

    all_indices = np.arange(len(train_dataset_for_split))
    train_labels = np.asarray([label for _, label in train_dataset_for_split.samples])
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_FRACTION,
        stratify=train_labels,
        random_state=RANDOM_SEED,
    )
    training_subset = Subset(train_dataset_for_augmentation, train_indices.tolist())
    validation_subset = Subset(train_dataset_for_split, validation_indices.tolist())

    training_labels = train_labels[train_indices]
    train_batch_sampler = BalancedClassBatchSampler(
        training_labels, SAMPLES_PER_CLASS_PER_BATCH, RANDOM_SEED
    )
    print(
        "Effective samples per balanced batch: "
        f"{dict.fromkeys(LABELS, SAMPLES_PER_CLASS_PER_BATCH)}"
    )

    train_loader = DataLoader(
        training_subset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = ExpressionCNN(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_state = None
    best_validation_macro_f1 = -1.0
    epochs_without_improvement = 0
    train_losses = []
    validation_macro_f1_scores = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        validation_accuracy, y_validation, y_validation_pred = accuracy_and_predictions(
            model, validation_loader, device
        )
        validation_macro_f1 = f1_score(
            y_validation, y_validation_pred, average="macro", zero_division=0
        )
        scheduler.step(validation_macro_f1)
        train_losses.append(average_loss)
        validation_macro_f1_scores.append(validation_macro_f1)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={average_loss:.4f} | "
            f"validation_accuracy={validation_accuracy:.4f} | "
            f"validation_macro_f1={validation_macro_f1:.4f}"
        )

        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_macro_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation accuracy stopped improving.")
                break

    model.load_state_dict(best_state)
    plot_history(train_losses, validation_macro_f1_scores)
    test_start = time.perf_counter()
    test_accuracy, y_test, y_pred = accuracy_and_predictions(model, test_loader, device)
    elapsed = time.perf_counter() - test_start

    print(f"\nBest validation macro F1: {best_validation_macro_f1:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(matrix)
    print(f"Average prediction time: {elapsed / len(test_dataset) * 1000.0:.4f} ms/image")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    figure, axis = plt.subplots(figsize=(9, 8))
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(figure)

    torch.save(
        {
            "architecture": "ExpressionCNN",
            "state_dict": model.state_dict(),
            "labels": LABELS,
            "input_size": INPUT_SIZE,
            "normalization_mean": NORMALIZATION_MEAN,
            "normalization_std": NORMALIZATION_STD,
            "best_validation_macro_f1": best_validation_macro_f1,
            "test_accuracy": test_accuracy,
        },
        MODEL_PATH,
    )
    print(f"Saved CNN model: {MODEL_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Saved training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
