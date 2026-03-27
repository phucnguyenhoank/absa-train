import torch

B = 2
n_aspect = 4
n_sentiment = 3
outputs = torch.randn(size=(B, n_aspect, n_sentiment))
print(outputs)

probs = torch.sigmoid(outputs)
print(probs)

max_probs, _ = torch.max(probs, dim=-1)
print(max_probs)

top2_values, top2_indices = torch.topk(max_probs, k=2, dim=-1)
print(top2_values)
print(top2_indices)
