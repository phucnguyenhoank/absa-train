import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERTMultiHead(nn.Module):

    def __init__(self, backbone_model_name, num_aspects, num_sentiments):
        super().__init__()

        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments

        self.phobert = AutoModel.from_pretrained(backbone_model_name)
        # freeze backbone
        for param in self.phobert.parameters():
            param.requires_grad = False

        # # unfreeze 2 last layers of PhoBERT
        # for param in self.phobert.encoder.layer[-2:].parameters():
        #     param.requires_grad = True

        hidden_size = self.phobert.config.hidden_size
        self.attentions = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_aspects)]
        )
        mid_dim = 32
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, mid_dim),
                    nn.BatchNorm1d(mid_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(mid_dim, num_sentiments),
                )
                for _ in range(num_aspects)
            ]
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.phobert(
            input_ids=input_ids, attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state  # (B, T, H) e.g. [64, 10, 768]

        logits = []

        for attn, classifier in zip(self.attentions, self.classifiers):

            # compute attention score for each token
            scores = attn(hidden).squeeze(-1)  # (B, T)

            # mask padding tokens
            scores = scores.masked_fill(attention_mask == 0, -1e9)

            # normalize
            weights = torch.softmax(scores, dim=1)  # (B, T)

            # weighted sum of token embeddings
            aspect_embedding = torch.sum(
                hidden * weights.unsqueeze(-1), dim=1
            )  # (B, H)

            # list of (B, num_sentiments)
            logits.append(classifier(aspect_embedding))

        logits = torch.stack(logits, dim=1)

        return logits  # (batch, aspects=4, sentiments=4)


class MultiHeadSigmoid(nn.Module):
    def __init__(self, backbone_model_name, num_aspects=4, num_sentiments=3):
        """
        Args:
            num_aspects: 4 (lecturer, training_program, facility, others)
            num_sentiments: 3 (negative, neutral, positive)
        """
        super().__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments

        self.backbone = AutoModel.from_pretrained(backbone_model_name)

        # Đóng băng backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # Mỗi aspect có 1 Attention head riêng để tìm keyword quan trọng
        self.aspect_attentions = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_aspects)]
        )

        mid_dim = 64
        self.sentiment_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, mid_dim),
                    nn.BatchNorm1d(mid_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(mid_dim, num_sentiments),  # Output: 3 nodes
                )
                for _ in range(num_aspects)
            ]
        )

    def forward(self, input_ids, attention_mask):
        # Lấy hidden states từ PhoBERT
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = (
            outputs.last_hidden_state
        )  # Shape: (Batch, Seq_Len, Hidden_Size)

        all_aspect_logits = []

        for i in range(self.num_aspects):
            # 1. Tính Attention cho aspect thứ i
            # (B, T, H) -> (B, T, 1) -> (B, T)
            attn_scores = self.aspect_attentions[i](hidden).squeeze(-1)

            # Masking các token padding
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

            # Chuẩn hóa trọng số attention
            weights = torch.softmax(attn_scores, dim=1)  # (B, T)

            # 2. Tổng hợp vector đại diện cho aspect (Weighted Sum)
            # (B, T, H) * (B, T, 1) -> sum theo T -> (B, H)
            aspect_repr = torch.sum(hidden * weights.unsqueeze(-1), dim=1)

            # 3. Phân loại sentiment cho aspect đó
            # Output shape: (B, 3) đại diện cho [Neg, Neu, Pos]
            logit = self.sentiment_classifiers[i](aspect_repr)
            all_aspect_logits.append(logit)

        # Gộp lại thành tensor duy nhất: (Batch, 4, 3)
        logits = torch.stack(all_aspect_logits, dim=1)

        return logits


class SimpleMultiHeadSigmoid(nn.Module):
    def __init__(self, backbone_model_name, num_aspects=4, num_sentiments=3):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments

        # 1. Load and Freeze Backbone
        self.backbone = AutoModel.from_pretrained(backbone_model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # 2. Minimal Linear Head: (Hidden_Size -> 4 * 3)
        # This maps the sentence embedding directly to all 12 sentiment nodes
        self.classifier = nn.Linear(hidden_size, num_aspects * num_sentiments)

    def forward(self, input_ids, attention_mask):
        # Get hidden states
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Use the [CLS] token (index 0) as the sentence representation
        # Shape: (Batch, Hidden_Size)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Single linear pass
        # Shape: (Batch, 12)
        logits = self.classifier(cls_output)

        # Reshape to match the original model's output: (Batch, 4, 3)
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)

        return logits
