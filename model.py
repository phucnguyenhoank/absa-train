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
        mid_dim = 128
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
