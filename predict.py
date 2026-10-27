import glob
import os
import torch
import matplotlib.pyplot as plt

from model import CNN
from preprocessor import load_image

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

files = sorted(glob.glob("my_digits/*.png"))

fig, axes = plt.subplots(1, len(files), figsize=(2 * len(files), 2.5))
if len(files) == 1:
    axes = [axes]

correct = 0
for ax, path in zip(axes, files):
    true_label = int(os.path.basename(path).split("_")[0])   # "8_0.png" -> 8

    img = load_image(path)              # (1, 28, 28), values 0..1
    x = img.unsqueeze(0)                # (1, 1, 28, 28) batch dimension

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    conf, pred = probs.max(dim=1)
    pred, conf = pred.item(), conf.item()

    correct += (pred == true_label)
    print(f"{os.path.basename(path)}: true={true_label} pred={pred} conf={conf:.2%}")

    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"{true_label}→{pred}\n{conf:.0%}", color="green" if pred == true_label else "red")
    ax.axis("off")

print(f"Accuracy: {correct}/{len(files)}")
plt.show()