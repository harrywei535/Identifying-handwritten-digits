import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import CNN


# --------------------
# 1. Load the datasets
# --------------------

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)


# --------------------
# 2. Create DataLoaders
# --------------------

batch_size = 100

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# --------------------
# 3. Create the CNN
# --------------------

model = CNN()


# --------------------
# 4. Loss function
# --------------------

criterion = nn.CrossEntropyLoss()


# --------------------
# 5. Optimizer
# --------------------

learning_rate = 0.001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# --------------------
# 6. Train the model
# --------------------

num_epochs = 10

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

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# --------------------
# 7. Training accuracy
# --------------------

model.eval()

train_correct = 0
train_total = 0

with torch.no_grad():

    for images, labels in train_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (predicted == labels).sum().item()


train_accuracy = 100 * train_correct / train_total

print(f"Training Accuracy: {train_accuracy:.2f}%")


# --------------------
# 8. Testing accuracy
# --------------------

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()


test_accuracy = 100 * test_correct / test_total

print(f"Testing Accuracy: {test_accuracy:.2f}%")


# --------------------
# 9. Save the model
# --------------------

torch.save(model.state_dict(), "mnist_cnn.pth")

print("Model saved!")