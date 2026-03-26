import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

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
# Class counts: Counter({3: 34840, 0: 8166, 1: 2201, 2: 497})


def compute_metrics(preds, labels):

    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    cm = confusion_matrix(labels, preds)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    print(classification_report(labels, preds))

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
    }


all_preds = []
all_labels = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())

        if i % 10 == 0:
            print(f"Batch {i}/{len(test_loader)} processed")


preds = torch.cat(all_preds)
labels = torch.cat(all_labels)

metrics = compute_metrics(preds, labels)
print(metrics)
