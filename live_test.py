import torch

from load import load_eval_model
from preprocess import tokenizer, rdrsegmenter
from config import topic_map, idx2sentiment

check_point = "absa-training-368da43_vnsf-44.pth"
model = load_eval_model(check_point)

test_sentence = "thầy dạy chi tiết , giảng giải kỹ ."

dot_separated_segmented_sentences = rdrsegmenter.word_segment(test_sentence)
test_segmented_sentence = " ".join(dot_separated_segmented_sentences)


test_tokenized_sentence = tokenizer(
    test_segmented_sentence, return_tensors="pt"
)
print(test_tokenized_sentence)
logits = model(**test_tokenized_sentence)

preds = logits.argmax(dim=-1)
pred = preds[0]

for aspect_idx, sentiment_idx_tensor in enumerate(pred):
    sentiment_idx = sentiment_idx_tensor.item()

    aspect = topic_map[aspect_idx]
    sentiment = idx2sentiment[sentiment_idx]

    print(f"{aspect}: {sentiment}")
