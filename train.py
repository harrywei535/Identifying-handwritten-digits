import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt


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

batch_size = 60

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
# 3. Define the neural network
# --------------------

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        # self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
        # x = self.layer2(x)
        # x = self.relu(x)
        x = self.layer3(x)

        return x


model = NeuralNetwork()


# --------------------
# 4. Loss function
# --------------------

criterion = nn.CrossEntropyLoss()


# --------------------
# 5. Optimizer
# --------------------

learning_rate = 0.0005

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# --------------------
# 6. Train the model
# --------------------

num_epochs = 10

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        # [100, 1, 28, 28] → [100, 784]
        images = images.view(-1, 28 * 28)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()

    # Print progress
    #if (epoch + 1) % 1 == 0:

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "f"Loss: {loss.item():.4f}"
        #f"Step [{i + 1}/{len(train_loader)}], "
        #f"Loss: {loss.item():.4f}"
    )


# --------------------
# 7. Test the model
# --------------------
model.eval()

train_correct = 0
train_total = 0

with torch.no_grad():
    for images, labels in train_loader:
        images = images.view(-1, 28 * 28)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

train_accuracy = 100 * train_correct / train_total

print(f"Training Accuracy: {train_accuracy:.2f}%")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        # Flatten images
        images = images.view(-1, 28 * 28)

        # Get scores
        outputs = model(images)

        # Get predicted digit
        _, predicted = torch.max(outputs, 1)

        # Count images
        total += labels.size(0)

        # Count correct predictions
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"Accuracy: {accuracy:.2f}%")


# # --------------------
# # 8. Display some predictions
# # --------------------

# images, labels = next(iter(test_loader))

# images_flat = images.view(-1, 28 * 28)

# with torch.no_grad():

#     outputs = model(images_flat)

#     _, predicted = torch.max(outputs, 1)


# for i in range(10):

#     plt.imshow(images[i].squeeze(), cmap="gray")

#     plt.title(
#         f"Actual: {labels[i].item()} | "
#         f"Predicted: {predicted[i].item()}"
#     )

#     plt.axis("off")
#     plt.show()