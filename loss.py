import torch
import torch.nn as nn


class AspectSentimentLoss(nn.Module):
    def __init__(self, device, aspect_weight=1.0, sentiment_weight=1.0):
        super().__init__()
        self.aspect_weight = aspect_weight
        self.sentiment_weight = sentiment_weight

        # lecturer, program, facility, others
        aspect_weights = torch.tensor([1.5, 2.0, 1.5, 4.5]).to(device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=aspect_weights)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        aspect_logits,  # (B, 4)
        sentiment_logits,  # (B, 4, 3)
        aspect_labels,  # (B, 4)
        sentiment_labels,  # (B, 4)
    ):
        # =========================
        # Aspect loss
        # =========================
        aspect_loss = self.bce(aspect_logits, aspect_labels.float())

        # =========================
        # Sentiment loss
        # =========================
        sentiment_loss = self.ce(
            sentiment_logits.view(-1, 3), sentiment_labels.view(-1)
        )

        # =========================
        # Total
        # =========================
        total_loss = (
            self.aspect_weight * aspect_loss
            + self.sentiment_weight * sentiment_loss
        )

        return {
            "loss": total_loss,
            "aspect_loss": aspect_loss.detach(),
            "sentiment_loss": sentiment_loss.detach(),
        }
