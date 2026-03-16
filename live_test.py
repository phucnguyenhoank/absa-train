from load import load_model
from preprocess import tokenizer, rdrsegmenter
from config import idx2topic, idx2sentiment

check_point = "absa-training-18833cf_vnsf-44_norm_dropout.pth"
model = load_model(check_point)
model.eval()

# sample = test_dataset[0]

# print("RAW SAMPLE:")
# print(sample)
# print()

# input_ids = sample["input_ids"].unsqueeze(0)
# attention_mask = sample["attention_mask"].unsqueeze(0)
# labels = sample["labels"]

# # Decode sentence
# decoded_sentence = tokenizer.decode(sample["input_ids"])
# print("Sentence:")
# print(decoded_sentence)
# print()

# print("GROUND TRUTH:")
# for aspect_idx, sentiment_idx in enumerate(labels):
#     aspect = idx2topic[aspect_idx]
#     sentiment = idx2sentiment[int(sentiment_idx)]
#     print(f"{aspect}: {sentiment}")

# print()

# # Model prediction
# model.eval()
# with torch.no_grad():
#     logits = model(input_ids=input_ids, attention_mask=attention_mask)
# print(logits)
# preds = logits.argmax(dim=-1)[0]

# print("PREDICTION:")
# print(preds)
# for aspect_idx, sentiment_idx_tensor in enumerate(preds):
#     sentiment_idx = sentiment_idx_tensor.item()

#     aspect = idx2topic[aspect_idx]
#     sentiment = idx2sentiment[sentiment_idx]

#     print(f"{aspect}: {sentiment}")

# ------------------------------------------


test_sentence = "phòng máy xịn"

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
