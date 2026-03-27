import torch
from load import load_model
from preprocess import tokenizer, rdrsegmenter
from config import (
    idx2topic,
    idx2sentiment,
)
from config_test import TEST_MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(TEST_MODEL_NAME).to(DEVICE)
model.eval()

test_sentence = "thầy thân thiện, nhưng bài tập thì khó"

# 1. Preprocess & Tokenize
dot_separated_segmented_sentences = rdrsegmenter.word_segment(test_sentence)
test_segmented_sentence = " ".join(dot_separated_segmented_sentences)
print(f"Segmented: {test_segmented_sentence}")

inputs = tokenizer(test_segmented_sentence, return_tensors="pt").to(DEVICE)

# 2. Forward pass
with torch.no_grad():
    logits = model(**inputs)
    probs = torch.sigmoid(logits).squeeze(0)  # Shape: (4, 3)

print(probs)

# 3. Tìm sentiment mạnh nhất của mỗi aspect
max_probs, sentiment_indices = torch.max(probs, dim=-1)  # (4,)
print(max_probs)
print(sentiment_indices)

# 4. Lấy Top 2 aspect có xác suất cao nhất
top2_values, top2_indices = torch.topk(max_probs, k=2)

print("\n--- Detection Results ---")

# Aspect đầu tiên (Top 1) - Luôn lấy theo logic "có ít nhất 1"
top1_score = top2_values[0].item()
top1_aspect_idx = top2_indices[0].item()

print(
    f"{idx2topic[top1_aspect_idx]}: {idx2sentiment[sentiment_indices[top1_aspect_idx].item()]} (Score: {top1_score:.2f})"
)

# Aspect thứ hai (Top 2) - Chỉ lấy nếu thỏa mãn tỷ lệ >= 0.5
top2_score = top2_values[1].item()
top2_aspect_idx = top2_indices[1].item()
threshold = 0.5
if (top2_score / top1_score) >= threshold:
    print(
        f"{idx2topic[top2_aspect_idx]}: {idx2sentiment[sentiment_indices[top2_aspect_idx].item()]} (Score: {top2_score:.2f})"
    )
else:
    print(
        f"-> Aspect thứ 2 ({idx2topic[top2_aspect_idx]}) bị loại vì score quá thấp so với Top 1."
    )
