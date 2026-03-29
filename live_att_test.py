import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load import load_model
from preprocess import tokenizer, rdrsegmenter
from config import idx2topic, idx2sentiment
from config_test import TEST_MODEL_NAME

# =========================
# 1. Setup
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(TEST_MODEL_NAME).to(DEVICE)
model.eval()

test_sentence = "thầy dạy hay, có tâm với sinh viên."

# =========================
# 2. Preprocess
# =========================
segmented = " ".join(rdrsegmenter.word_segment(test_sentence))
print(f"Segmented: {segmented}")

inputs = tokenizer(segmented, return_tensors="pt").to(DEVICE)

# =========================
# 3. Inference
# =========================
with torch.no_grad():
    logits, attentions = model(**inputs, return_attentions=True)

probs = torch.sigmoid(logits).squeeze(0)  # (4, 3)
attentions = attentions.squeeze(0)  # (4, T)

print(probs)
print("Attention shape:", attentions.shape)

# =========================
# 4. Post-process (Top aspects)
# =========================
max_probs, sentiment_indices = torch.max(probs, dim=-1)  # (4,)

top2_values, top2_indices = torch.topk(max_probs, k=2)

print("\n--- Detection Results ---")

# Top 1
top1_score = top2_values[0].item()
top1_idx = top2_indices[0].item()

print(
    f"{idx2topic[top1_idx]}: "
    f"{idx2sentiment[sentiment_indices[top1_idx].item()]} "
    f"(Score: {top1_score:.2f})"
)

# Top 2 (threshold)
top2_score = top2_values[1].item()
top2_idx = top2_indices[1].item()

threshold = 0.5
if (top2_score / top1_score) >= threshold:
    print(
        f"{idx2topic[top2_idx]}: "
        f"{idx2sentiment[sentiment_indices[top2_idx].item()]} "
        f"(Score: {top2_score:.2f})"
    )
else:
    print(
        f"-> Aspect thứ 2 ({idx2topic[top2_idx]}) "
        f"bị loại vì score quá thấp so với Top 1."
    )

# =========================
# 5. Visualization - Bar Chart
# =========================
probs_np = probs.cpu().numpy()

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(4)
width = 0.25

labels = ["Negative", "Neutral", "Positive"]
colors = ["#e74c3c", "#f1c40f", "#2ecc71"]

for i in range(3):
    ax.bar(
        x + i * width,
        probs_np[:, i],
        width,
        label=labels[i],
        color=colors[i],
    )

ax.set_xticks(x + width)
ax.set_xticklabels(["Lecturer", "Program", "Facility", "Others"])

ax.set_ylabel("Probability")
ax.set_title("Sentiment Distribution per Aspect")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()

# =========================
# 6. Visualization - Attention Heatmap
# =========================
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(12, 4))

sns.heatmap(
    attentions.cpu().numpy(),
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor="gray",
)

plt.yticks(
    range(4),
    ["Lecturer", "Program", "Facility", "Others"],
    rotation=0,
)

plt.xticks(
    range(len(tokens)),
    tokens,
    rotation=45,
    ha="right",
)

plt.title("Attention Heatmap per Aspect")

plt.tight_layout()
plt.show()
