import pandas as pd
from datasets import Dataset, DatasetDict
import ast

from config import sentiment2idx, idx2topic
from preprocess import tokenizer, rdrsegmenter


def load_and_fix_dataset(file_path):
    # 1. Load with pandas (it will treat [0, 1] as a string automatically)
    df = pd.read_csv(file_path)

    # 2. Convert the string "[0, 1]" into an actual Python list [0, 1]
    # We use ast.literal_eval because it's safer than eval()
    df["topic"] = df["topic"].apply(lambda x: ast.literal_eval(str(x)))
    df["sentiment"] = df["sentiment"].apply(lambda x: ast.literal_eval(str(x)))
    return Dataset.from_pandas(df)


# Load all splits
ds = DatasetDict(
    {
        "train": load_and_fix_dataset(
            "multisentiment-uit-vsfc/df_final_train.csv"
        ),
        "validation": load_and_fix_dataset(
            "multisentiment-uit-vsfc/df_final_validation.csv"
        ),
        "test": load_and_fix_dataset(
            "multisentiment-uit-vsfc/df_final_test.csv"
        ),
    }
)
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
    # Tokenize câu đầu vào
    model_inputs = tokenizer(
        examples["sentence"],
        padding=False,
        truncation=True,
        max_length=256,
    )

    all_labels = []

    # Giả sử:
    # examples["topic"] là list của các list: [[0, 2], [1], ...]
    # examples["sentiment"] là list của các list: [[2, 0], [1], ...] (đã bỏ nhãn 'none')
    for topics, sentiments in zip(examples["topic"], examples["sentiment"]):

        # 1. Tạo ma trận nhãn toàn số 0: (num_aspects=4, num_sentiments=3)
        # Hàng: Lecturer, Program, Facility, Others
        # Cột: Neg, Neu, Pos
        label_matrix = [[0.0] * 3 for _ in range(4)]

        # 2. Điền số 1 vào đúng vị trí Aspect + Sentiment xuất hiện
        for t, s in zip(topics, sentiments):
            topic_idx = int(t)
            sentiment_idx = int(s)
            label_matrix[topic_idx][sentiment_idx] = 1.0

        all_labels.append(label_matrix)

    model_inputs["labels"] = all_labels
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
