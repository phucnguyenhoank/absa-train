from datasets import load_from_disk, load_dataset

from config import sentiment2idx, idx2topic
from preprocess import tokenizer, rdrsegmenter

ds = load_from_disk("uit-vsfc")
print(ds["test"][0])


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


# rm -rf ~/.cache/huggingface/datasets, or load_from_cache_file=False
ds_segmented = ds.map(segment_text)
print(ds_segmented["test"][0])


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
        aspect_labels = [sentiment2idx["none"]] * len(idx2topic)
        aspect_labels[topic] = sentiment
        labels.append(aspect_labels)

    model_inputs["labels"] = labels
    return model_inputs


tokenized_datasets = ds_segmented.map(preprocess_function, batched=True)
print(tokenized_datasets["test"][0])


# Convert to PyTorch format
pt_datasets = tokenized_datasets.remove_columns(
    ["sentence", "sentiment", "topic"]
)

pt_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)
print(pt_datasets["test"][0])


train_dataset = pt_datasets["train"]
val_dataset = pt_datasets["validation"]
test_dataset = pt_datasets["test"]
