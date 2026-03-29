import torch
from torch.utils.data import DataLoader, Subset
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

from config import BATCH_SIZE, SUBSET_SIZE
from data import test_dataset
from load import load_model
from preprocess import tokenizer
from config_test import TEST_MODEL_NAME, history_path


# === Create output directory ===
output_dir = f"results_{TEST_MODEL_NAME}"
os.makedirs(output_dir, exist_ok=True)


# === LOSS HISTORY VISUALIZATION ===
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
loss_path = os.path.join(output_dir, "train_vs_val_loss.png")
plt.savefig(loss_path)
plt.close()
print("Plot saved!")


# === METRICS CALCULATION ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(TEST_MODEL_NAME)
print("Model loaded!")

collator = DataCollatorWithPadding(tokenizer)

# test_dataset = Subset(test_dataset, range(SUBSET_SIZE))

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collator,
)


def save_text(text, filename):
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def compute_metrics(preds, labels):
    """
    preds: (N, 4) - Giá trị 0, 1, 2 hoặc 3 (none)
    labels: (N, 4) - Giá trị 0, 1, 2 hoặc 3 (none)
    """
    preds_np = preds.numpy()
    labels_np = labels.numpy()
    topic_names = ["lecturer", "program", "facility", "others"]
    sentiment_names = ["negative", "neutral", "positive"]

    # --- 1. ASPECT DETECTION METRICS (NHỊ PHÂN) ---
    # Coi như bài toán tìm đúng aspect (1) hay không (0)
    y_true_det = (labels_np < 3).astype(int)
    y_pred_det = (preds_np < 3).astype(int)

    print("\n" + "=" * 20 + " 1. ASPECT DETECTION " + "=" * 20)

    for i in range(4):
        binary_names = [f"not_{topic_names[i]}", topic_names[i]]
        aspect_report = classification_report(
            y_true_det[:, i],
            y_pred_det[:, i],
            labels=[0, 1],
            target_names=binary_names,
            digits=4,
        )
        cm = confusion_matrix(
            y_true_det[:, i],
            y_pred_det[:, i],
            labels=[0, 1],
        )
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=binary_names,
            yticklabels=binary_names,
            cmap="Blues",
        )

        plt.title(f"Confusion Matrix - {topic_names[i]}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.savefig(
            os.path.join(output_dir, f"aspect_cm_{topic_names[i]}.png")
        )
        plt.close()

        print(f"\nAspect for {topic_names[i]}:")
        print(aspect_report)
        save_text(aspect_report, f"aspect_{topic_names[i]}_detection.txt")

    aspect_report = classification_report(
        y_true_det, y_pred_det, target_names=topic_names, digits=4
    )
    print(aspect_report)
    save_text(aspect_report, "aspect_detection.txt")

    # --- 2. SENTIMENT CLASSIFICATION METRICS (CHỈ TÍNH TRÊN TRUE POSITIVES) ---
    print(
        "\n"
        + "=" * 20
        + " 2. SENTIMENT (ON TRUE POSITIVE ASPECTS) "
        + "=" * 20
    )

    all_s_true = []
    all_s_pred = []

    for i in range(4):
        # Chỉ lấy những mẫu mà cả GT và Pred đều tìm thấy Aspect i
        tp_mask = (labels_np[:, i] < 3) & (preds_np[:, i] < 3)

        if tp_mask.any():
            s_true = labels_np[tp_mask, i]
            s_pred = preds_np[tp_mask, i]
            all_s_true.extend(s_true)
            all_s_pred.extend(s_pred)

            # In báo cáo nhỏ cho từng aspect
            report = classification_report(
                s_true, s_pred, target_names=sentiment_names
            )
            print(f"\nSentiment for {topic_names[i]}:")
            print(report)
            save_text(report, f"sentiment_{topic_names[i]}.txt")

    # Báo cáo tổng hợp cảm xúc trên toàn bộ các aspect bắt đúng
    if all_s_true:
        sent_report = classification_report(
            all_s_true, all_s_pred, target_names=sentiment_names, digits=4
        )
        print(f"\nSentiment for all aspects:")
        print(sent_report)
        save_text(sent_report, "sentiment_overall.txt")

        # Vẽ Confusion Matrix cho Sentiment (Soi lỗi Neutral ở đây)
        cm = confusion_matrix(all_s_true, all_s_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=sentiment_names,
            yticklabels=sentiment_names,
            cmap="Blues",
        )
        plt.title("Sentiment Confusion Matrix (TP Aspects only)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(output_dir, "sentiment_cm.png"))
        plt.close()

    return {
        "aspect_f1_micro": f1_score(y_true_det, y_pred_det, average="micro"),
        "sentiment_f1_macro": (
            f1_score(all_s_true, all_s_pred, average="macro")
            if all_s_true
            else 0
        ),
    }


all_preds = []
all_labels = []
ratio_threshold = 0.5

with torch.no_grad():
    model.eval()
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        probs = torch.sigmoid(logits)
        max_vals, max_indices = torch.max(probs, dim=-1)

        batch_size = logits.size(0)
        final_preds = torch.full((batch_size, 4), 3, device=DEVICE)

        for b in range(batch_size):
            top2_vals, top2_idxs = torch.topk(max_vals[b], k=2)
            # Luôn lấy Top 1
            final_preds[b, top2_idxs[0]] = max_indices[b, top2_idxs[0]]

            # Xét Top 2 theo tỷ lệ
            if (top2_vals[1] / top2_vals[0]) >= ratio_threshold:
                final_preds[b, top2_idxs[1]] = max_indices[b, top2_idxs[1]]

        # Ground truth mapping
        gt_max_vals, gt_indices = torch.max(batch["labels"], dim=-1)
        final_labels = torch.full((batch_size, 4), 3, device=DEVICE)
        gt_mask = gt_max_vals > 0
        final_labels[gt_mask] = gt_indices[gt_mask]

        all_preds.append(final_preds.cpu())
        all_labels.append(final_labels.cpu())

# Kết quả cuối
preds_tensor = torch.cat(all_preds)
labels_tensor = torch.cat(all_labels)
metrics = compute_metrics(preds_tensor, labels_tensor)
print(TEST_MODEL_NAME)
