import torch
from collections import Counter


def adapt_shape(outputs, labels):

    B, A, C = outputs.shape

    logits = outputs.view(B * A, C)
    targets = labels.view(B * A)

    return logits, targets


def calculate_alpha(dataset, num_classes=4, device="cpu"):

    all_labels = []

    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        all_labels.extend(labels.tolist())

    counts = Counter(all_labels)
    total = sum(counts.values())

    alpha = []

    for i in range(num_classes):

        class_count = counts.get(i, 0)

        weight = 1.0 - (class_count / total)

        alpha.append(weight)

    return torch.tensor(alpha).to(device), counts
