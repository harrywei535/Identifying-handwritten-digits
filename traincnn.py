import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from model import CNN
from preprocessor import deskew_image


def deskew(img):
    """Deskew an MNIST PIL image so training matches what preprocessor.py feeds the model."""
    a = deskew_image(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.clip(a, 0.0, 1.0)).unsqueeze(0)


# Training images get random rotation, shift, scale and shear so the model
# copes with digits that aren't perfectly MNIST-like (photos, other pens)
train_transform = transforms.Compose([
    transforms.Lambda(deskew),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.8, 1.15),
        shear=10,
    ),
])

# Validation and test images are only deskewed, never augmented
eval_transform = transforms.Lambda(deskew)


def main():

    # --------------------
    # 1. Load the datasets
    # --------------------

    # Two copies of the training set so the validation split can use
    # different (non-augmented) transforms from the training split
    train_full = datasets.MNIST(root="./data", train=True, transform=train_transform, download=True)
    val_full = datasets.MNIST(root="./data", train=True, transform=eval_transform, download=True)

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=eval_transform,
        download=True
    )

    # Hold out 6,000 training images for validation (fixed seed = same split every run)
    val_size = 6000
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42))
    train_dataset = Subset(train_full, perm[val_size:].tolist())
    val_dataset = Subset(val_full, perm[:val_size].tolist())


    # --------------------
    # 2. Create DataLoaders
    # --------------------

    batch_size = 100
    workers = 2    # augmentation runs on the CPU, so do it in background processes

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )


    # --------------------
    # 3. Create the CNN
    # --------------------

    torch.manual_seed(0)
    model = CNN()


    # --------------------
    # 4. Loss function
    # --------------------

    # Label smoothing stops the model from being 100% sure of everything,
    # which makes its confidence on odd-looking photos more honest
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)


    # --------------------
    # 5. Optimizer
    # --------------------

    learning_rate = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Halve the learning rate when validation accuracy stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )


    # --------------------
    # 6. Train the model, keeping the best checkpoint
    # --------------------

    # Optional: python traincnn.py 5  -> train for 5 epochs
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    checkpoint = "mnist_cnn.pth"
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)
        scheduler.step(val_accuracy)

        # Save only when the model beats its previous best on held-out data,
        # so a later epoch that overfits can't overwrite a better model
        marker = ""
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch + 1
            torch.save(model.state_dict(), checkpoint)
            marker = "  (best, saved)"

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {average_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}"
        )


    # --------------------
    # 7. Testing accuracy (on the best checkpoint, not the last epoch)
    # --------------------

    model.load_state_dict(torch.load(checkpoint))

    test_accuracy = evaluate(model, test_loader)

    print(f"Best epoch: {best_epoch}, Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")
    print(f"Model saved to {checkpoint}!")


def evaluate(model, loader):
    """Accuracy (%) of `model` on every image in `loader`."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# Windows starts DataLoader workers as new processes that re-import this
# file, so the training code must only run when the script is run directly
if __name__ == "__main__":
    main()