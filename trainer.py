import torch


def focal_binary_cross_entropy(logits, targets, gamma=2.0, alpha=1.0):
    """
    Focal Loss for multi-label classification.
    logits: (B, 4, 3)
    targets: (B, 4, 3)
    """
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (
        1 - targets
    )  # probability of the correct class

    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    # The Focal factor (1 - pt)^gamma
    focal_weight = (1 - p_t) ** gamma

    loss = alpha * focal_weight * bce_loss

    return loss.mean()


def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()

    total_loss = 0

    for batch in dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # loss = focal_binary_cross_entropy(outputs, batch["labels"])
        loss = criterion(outputs, batch["labels"])

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, criterion):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = criterion(outputs, batch["labels"])

            total_loss += loss.item()

    return total_loss / len(dataloader)
