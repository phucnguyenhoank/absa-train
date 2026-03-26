from load import load_model
from preprocess import tokenizer, rdrsegmenter
from config import idx2topic, idx2sentiment
from config_test import TEST_MODEL_NAME

model = load_model(TEST_MODEL_NAME)
model.eval()

test_sentence = "chương trình dạy không sát thực tế, lan man"

dot_separated_segmented_sentences = rdrsegmenter.word_segment(test_sentence)
test_segmented_sentence = " ".join(dot_separated_segmented_sentences)
print(test_segmented_sentence)

test_tokenized_sentence = tokenizer(
    test_segmented_sentence, return_tensors="pt"
)

logits = model(**test_tokenized_sentence)
preds = logits.argmax(dim=-1)
pred = preds[0]

for aspect_idx, sentiment_idx_tensor in enumerate(pred):
    sentiment_idx = sentiment_idx_tensor.item()

    aspect = idx2topic[aspect_idx]
    sentiment = idx2sentiment[sentiment_idx]

    print(f"{aspect}: {sentiment}")
