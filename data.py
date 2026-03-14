import os
import py_vncorenlp

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from config import backbone_model_name, sentiment2idx, topic_map

ds = load_from_disk("uit-vsfc")

cwd = Path.cwd()

# Define the ABSOLUTE path
save_dir = (cwd / "vncorenlp").resolve()

# Manually create the parent directory first
# The library's download_model fails because it can't create /vncorenlp/models
# if /vncorenlp doesn't exist yet.
save_dir.mkdir(parents=True, exist_ok=True)

if not (save_dir / "models").exists():
    print(f"Downloading models to {save_dir}...")
    # The command below tried to run a command like: Create folder "models" inside "vncorenlp"
    # BUT it does not create the "vncorenlp" folder.
    # VERY VERY FRUSTRATING
    py_vncorenlp.download_model(save_dir=save_dir.as_posix())
else:
    print("Models already exist, skipping download.")


try:
    rdrsegmenter = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"], save_dir=save_dir.as_posix()
    )
    print("Segmenter loaded successfully!")
except Exception as e:
    print(f"Error loading segmenter: {e}")

os.chdir(cwd)
print(Path.cwd())


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
    # the sentence result will be separated.
    dot_separated_segmented_sentences = rdrsegmenter.word_segment(sentence)

    # Join back into a single string with spaces
    example["sentence"] = " ".join(dot_separated_segmented_sentences)
    return example


ds_segmented = ds.map(segment_text)


tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)


def preprocess_function(examples):
    # IMPORTANT: Ensure examples["sentence"] is ALREADY word-segmented

    model_inputs = tokenizer(
        examples["sentence"],
        padding=False,
        truncation=True,
        max_length=256,
    )

    labels = []
    for topic, sentiment in zip(examples["sentiment"], examples["topic"]):
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
