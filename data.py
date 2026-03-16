from datasets import load_from_disk

from config import sentiment2idx, topic_map
from preprocess import tokenizer, rdrsegmenter

ds = load_from_disk("uit-vsfc")


def segment_text(example):
    """
    Input:
        [
            "môn học này rất tuyệt, vừa học được lý thuyết.",
            "thầy dạy chậm rãi."
        ]
    Output:
        [
            "môn_học này rất tuyệt , vừa học được lý_thuyết .",
            "thầy dạy chậm_rãi ."
        ]
    """
    sentence = example["sentence"]

    # if the input sentence contains ".",
    # the sentence result will be separated into a list.
    dot_separated_segmented_sentences = rdrsegmenter.word_segment(sentence)

    # Join the list of sentences back into a single string with spaces
    example["sentence"] = " ".join(dot_separated_segmented_sentences)
    return example


ds_segmented = ds.map(segment_text)


def preprocess_function(examples):
    # IMPORTANT: Ensure examples["sentence"] is ALREADY word-segmented

    model_inputs = tokenizer(
        examples["sentence"],
        padding=False,
        truncation=True,
        max_length=256,
    )

    labels = []
    for topic, sentiment in zip(examples["topic"], examples["sentiment"]):
        aspect_labels = [sentiment2idx["none"]] * len(topic_map)
        aspect_labels[topic] = sentiment
        labels.append(aspect_labels)

    model_inputs["labels"] = labels
    return model_inputs


tokenized_datasets = ds_segmented.map(preprocess_function, batched=True)

# Convert to PyTorch format
pt_datasets = tokenized_datasets.remove_columns(
    ["sentence", "sentiment", "topic"]
)

pt_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)


train_dataset = pt_datasets["train"]
val_dataset = pt_datasets["validation"]
test_dataset = pt_datasets["test"]
