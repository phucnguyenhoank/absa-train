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

        # unfreeze 3 last layers of PhoBERT
        for param in self.phobert.encoder.layer[-3:].parameters():
            param.requires_grad = True

        hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_size)
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

        cls_output = outputs.last_hidden_state[:, 0]

        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)

        logits = [classifier(cls_output) for classifier in self.classifiers]

        logits = torch.stack(logits, dim=1)

        return logits  # (batch, aspects=4, sentiments=4)
