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
        class_count = counts.get(i, 1)  # avoid division by 0
        weight = 1.0 / class_count
        alpha.append(weight)

    alpha = torch.tensor(alpha, dtype=torch.float32)

    # normalize (so sum = num_classes or 1)
    alpha = alpha / alpha.sum()

    # manually reduce "none" class (class 3)
    alpha[3] *= 0.3  # tune (0.1 ~ 0.5 depending on how much ignorance)
    alpha = alpha / alpha.sum()

    return alpha.to(device), counts
