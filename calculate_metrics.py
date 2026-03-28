import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns

from config import BATCH_SIZE
from data import test_dataset
from load import load_model
from preprocess import tokenizer
from config_test import TEST_MODEL_NAME, history_path


# === Create output directory ===
output_dir = f"results_{TEST_MODEL_NAME}"
os.makedirs(output_dir, exist_ok=True)


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

# === Save plot to directory ===
loss_path = os.path.join(output_dir, "train_vs_val_loss.png")
plt.savefig(loss_path)
plt.close()
print("Plot saved!")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(TEST_MODEL_NAME)
print("Model loaded!")

collator = DataCollatorWithPadding(tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collator,
)


def compute_metrics(preds, labels):
    preds_flat = preds.reshape(-1).cpu().numpy()
    labels_flat = labels.reshape(-1).cpu().numpy()
    target_names = ["negative", "neutral", "positive", "none"]

    # === Classification report ===
    report = classification_report(
        labels_flat, preds_flat, target_names=target_names, digits=4
    )

    print("\n--- Classification Report ---")
    print(report)

    # === Save report ===
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # === Confusion matrix ===
    cm = confusion_matrix(labels_flat, preds_flat)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    return {
        "f1_macro": f1_score(labels_flat, preds_flat, average="macro"),
        "f1_micro": f1_score(labels_flat, preds_flat, average="micro"),
    }


all_preds = []
all_labels = []

ratio_threshold = 0.5
min_floor = 0.001

with torch.no_grad():
    model.eval()
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        probs = torch.sigmoid(logits)

        max_vals, max_indices = torch.max(probs, dim=-1)

        batch_size = logits.size(0)
        final_preds = torch.full((batch_size, 4), 3, device=DEVICE)

        for b in range(batch_size):
            top2_vals, top2_idxs = torch.topk(max_vals[b], k=2)

            if top2_vals[0] >= min_floor:
                a1_idx = top2_idxs[0]
                final_preds[b, a1_idx] = max_indices[b, a1_idx]

                if (top2_vals[1] / top2_vals[0]) >= ratio_threshold:
                    a2_idx = top2_idxs[1]
                    final_preds[b, a2_idx] = max_indices[b, a2_idx]

        gt_max_vals, gt_indices = torch.max(batch["labels"], dim=-1)
        final_labels = torch.full((batch_size, 4), 3, device=DEVICE)
        gt_mask = gt_max_vals > 0
        final_labels[gt_mask] = gt_indices[gt_mask]

        all_preds.append(final_preds.cpu())
        all_labels.append(final_labels.cpu())

preds_tensor = torch.cat(all_preds)
labels_tensor = torch.cat(all_labels)

print(f"\nModel: {TEST_MODEL_NAME}")
metrics = compute_metrics(preds_tensor, labels_tensor)
print(metrics)

print(f"\nAll outputs saved in: {output_dir}")
