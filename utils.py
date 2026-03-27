import torch
from collections import Counter


def calculate_alpha(dataset, num_classes=4, device="cpu"):
    all_labels = []
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        all_labels.extend(labels.tolist())

    class_counts = Counter(all_labels)
    alpha = []
    for i in range(num_classes):
        weight = 1.0 / class_counts[i]
        alpha.append(weight)

    alpha = torch.tensor(alpha, dtype=torch.float32)
    alpha = alpha / alpha.sum()
    return alpha.to(device), class_counts
