import matplotlib.pyplot as plt
import json

history_path = "loss_history.json"

with open(history_path) as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss = history["val_loss"]

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("train_vs_val_loss.png")
print("Plot saved!")
