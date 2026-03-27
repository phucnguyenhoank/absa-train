import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import BATCH_SIZE
from data import test_dataset
from load import load_model
from preprocess import tokenizer
from config_test import TEST_MODEL_NAME

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

    print("\n--- Classification Report ---")
    print(
        classification_report(
            labels_flat, preds_flat, target_names=target_names, digits=4
        )
    )
    return {
        "f1_macro": f1_score(labels_flat, preds_flat, average="macro"),
        "f1_micro": f1_score(labels_flat, preds_flat, average="micro"),
    }


all_preds = []
all_labels = []

# Tỷ lệ để chấp nhận aspect thứ 2
ratio_threshold = 0.5
# Ngưỡng sàn cực thấp để tránh lấy "rác" nếu câu hoàn toàn không có nghĩa
min_floor = 0.001

with torch.no_grad():
    model.eval()
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        probs = torch.sigmoid(logits)  # (Batch, 4, 3)

        # 1. Tìm sentiment mạnh nhất của mỗi aspect
        max_vals, max_indices = torch.max(probs, dim=-1)  # (Batch, 4)

        # 2. Khởi tạo nhãn dự đoán là 'none' (3)
        batch_size = logits.size(0)
        final_preds = torch.full((batch_size, 4), 3, device=DEVICE)

        # 3. Áp dụng logic Top 2 & Ratio cho từng sample trong batch
        for b in range(batch_size):
            # Lấy top 2 aspect có score cao nhất của sample b
            top2_vals, top2_idxs = torch.topk(max_vals[b], k=2)

            # --- Xử lý Top 1 ---
            if top2_vals[0] >= min_floor:
                a1_idx = top2_idxs[0]
                final_preds[b, a1_idx] = max_indices[b, a1_idx]

                # --- Xử lý Top 2 dựa trên tỷ lệ so với Top 1 ---
                if (top2_vals[1] / top2_vals[0]) >= ratio_threshold:
                    a2_idx = top2_idxs[1]
                    final_preds[b, a2_idx] = max_indices[b, a2_idx]

        # 4. Xử lý Ground Truth (giữ nguyên logic cũ)
        gt_max_vals, gt_indices = torch.max(batch["labels"], dim=-1)
        final_labels = torch.full((batch_size, 4), 3, device=DEVICE)
        gt_mask = gt_max_vals > 0
        final_labels[gt_mask] = gt_indices[gt_mask]

        all_preds.append(final_preds.cpu())
        all_labels.append(final_labels.cpu())

# Concatenate all batches
preds_tensor = torch.cat(all_preds)
labels_tensor = torch.cat(all_labels)

print(f"\nModel: {TEST_MODEL_NAME}")
metrics = compute_metrics(preds_tensor, labels_tensor)
print(metrics)
