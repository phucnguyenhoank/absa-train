import torch
from utils import adapt_shape
from loss import focal_loss


def train_epoch(model, dataloader, optimizer, alpha, device):
    model.train()

    total_loss = 0

    for batch in dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits, targets = adapt_shape(outputs, batch["labels"])

        loss = focal_loss(logits, targets, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, alpha, device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits, targets = adapt_shape(outputs, batch["labels"])

            loss = focal_loss(logits, targets, alpha=alpha)

            total_loss += loss.item()

    return total_loss / len(dataloader)
