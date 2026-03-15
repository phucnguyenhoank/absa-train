import sys

print(sys.version)

import os
import json

from transformers import DataCollatorWithPadding, AutoModel
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from config import *
from data import (
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer,
)

target_path = os.getenv("AIP_MODEL_DIR")
print(f"AIP_MODEL_DIR={target_path}")

run_mode = os.getenv("RUN_MODE")
print(f"run_mode={run_mode}")

if run_mode == "sanity_check":
    train_dataset = Subset(train_dataset, range(SUBSET_SIZE))
    val_dataset = Subset(val_dataset, range(SUBSET_SIZE))
    test_dataset = Subset(test_dataset, range(SUBSET_SIZE))

collator = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, collate_fn=collator
)


def calculate_alpha(dataset, num_classes=4, device="cpu"):
    all_labels = []

    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        all_labels.extend(labels.tolist())

    counts = Counter(all_labels)
    total = sum(counts.values())

    # Sort by class index to ensure [alpha0, alpha1, alpha2, alpha3]
    alpha = []
    for i in range(num_classes):
        # Inverse frequency: more frequent classes get lower weight
        class_count = counts.get(i, 0)
        weight = 1.0 - (class_count / total)
        alpha.append(weight)

    return torch.tensor(alpha).to(device), counts


class PhoBERTMultiHead(nn.Module):
    def __init__(self, backbone_model_name, num_aspects, num_sentiments):
        super().__init__()

        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments

        # PhoBERT backbone
        self.phobert = AutoModel.from_pretrained(backbone_model_name)

        # 4 heads
        hidden_size = self.phobert.config.hidden_size  # 768
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(hidden_size, num_sentiments)
                for _ in range(num_aspects)
            ]
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids, attention_mask=attention_mask
        )

        cls_output = outputs.pooler_output

        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(cls_output))

        # aspects of (batch, sentiments) -> (batch, aspects, sentiments)
        logits = torch.stack(logits, dim=1)
        return logits


def focal_loss(logits, targets, gamma=2, alpha=None):
    """
    logits=tensor([[-0.7488, -0.3002,  1.4807],
        [ 1.0669,  1.3075,  0.5109]])
    labels=tensor([0, 1])
    alpha = torch.tensor([1.0, 1.0, 1.0, 0.25])
    """
    ce_losses = F.cross_entropy(logits, targets, reduction="none")
    pts = torch.exp(-ce_losses)
    focal = (1 - pts) ** gamma * ce_losses
    if alpha is not None:
        focal = alpha[targets] * focal
    return focal.mean()


def adapt_shape(outputs, labels):
    # TODO: mask removing NONE labels.
    B, A, C = outputs.shape
    logits = outputs.view(B * A, C)
    targets = labels.view(B * A)
    return logits, targets


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")
model = PhoBERTMultiHead(
    backbone_model_name,
    num_aspects=len(topic_map),
    num_sentiments=len(idx2sentiment),
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

alpha_weights, class_counts = calculate_alpha(train_dataset, device=device)
print(f"Counts per class (not sorted): {class_counts}")
print(f"Recommended alpha: {alpha_weights}")

# Training

best_val_loss = float("inf")

train_losses = []
val_losses = []
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}")

    # -------------------
    # TRAIN
    # -------------------
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["labels"]
        logits, targets = adapt_shape(outputs, labels)
        loss = focal_loss(logits, targets, alpha=alpha_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print("Train loss:", train_loss)
    train_losses.append(train_loss)

    # -------------------
    # VALIDATION
    # -------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            labels = batch["labels"]
            logits, targets = adapt_shape(outputs, labels)
            loss = focal_loss(logits, targets, alpha=alpha_weights)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print("Validation loss:", val_loss)
    val_losses.append(val_loss)

    # -------------------
    # CHECKPOINT
    # -------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_aspects": model.num_aspects,
            "num_sentiments": model.num_sentiments,
            "backbone_model_name": backbone_model_name,
        }
        torch.save(checkpoint, f"{best_model_name}.pth")
        print(f"Best {best_model_name} saved")
    else:
        patience_counter += 1

    # -------------------
    # EARLY STOPPING
    # -------------------
    if patience_counter >= PATIENCE:
        print("Early stopping")
        break

history = {"train_loss": train_losses, "val_loss": val_losses}
history_path = "loss_history.json"
with open(history_path, "w") as f:
    json.dump(history, f)

print("DONE training")


if target_path:
    os.system(
        f"gsutil cp {best_model_name}.pth {target_path}/{best_model_name}.pth"
    )
    os.system(f"gsutil cp {history_path} {target_path}/{history_path}")
    print(f"Training result SAVED {target_path}!")
