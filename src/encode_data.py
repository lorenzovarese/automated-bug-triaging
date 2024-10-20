from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
import os, multiprocessing

MODEL_NAME = "bert-base-uncased"
CONTEXT_LENGTH = 512
NUM_PROC = min(100, multiprocessing.cpu_count() - 1)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    encoding  = TOKENIZER(examples["text"], padding="max_length", truncation=True, max_length=CONTEXT_LENGTH)
    encoding["label"] = examples["label"]
    return encoding

def encode_data(
        encoded_data_path=os.path.join("data", "encoded_data"), 
        data_path=os.path.join("data", "issues.json"), 
        only_recent=False,
        force=False,
        verbose=False,
    ):
    if only_recent:
        encoded_data_path = encoded_data_path + "_recent"

    if not force and os.path.exists(encoded_data_path):
        if verbose: print(f"Loading encoded data from '{encoded_data_path}'...")
        return datasets.load_from_disk(encoded_data_path)

    if verbose: 
        print(f"No cached data found at {encoded_data_path}.")
        print(f"Loading data from '{data_path}'...")

    issues_df = pd.read_json(data_path)

    if only_recent:
        if verbose: print("Filtering only recent data...")
        issues_df = issues_df[190_000 <= issues_df["github_id"]]

        # filter out issues with only one assignee because we can't split them
        assignees_counts = issues_df["assignee"].value_counts()
        if verbose: print(f"Filtering out {len(assignees_counts[assignees_counts == 1])} assignees with only one issue...")
        assignees_with_multiple_issues = assignees_counts[assignees_counts > 1].index
        issues_df = issues_df[issues_df["assignee"].isin(assignees_with_multiple_issues)]

    issues_df["label"] = issues_df["assignee"].astype("category").cat.codes

    train_df = issues_df[issues_df["github_id"] <= 210_000]
    train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])
    test_df = issues_df[(210_000 < issues_df["github_id"]) & (issues_df["github_id"] <= 220_000)]

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "eval": datasets.Dataset.from_pandas(eval_df),
        "test": datasets.Dataset.from_pandas(test_df),
    })
    
    if verbose: print("Encoding data...")
    encoded_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[column for column in dataset['train'].column_names if column not in ["label", "github_id"]], 
        # remove_columns=dataset['train'].column_names,
        num_proc=NUM_PROC
    )

    if verbose: print(f"Saving encoded data to {encoded_data_path}...")
    encoded_dataset.save_to_disk(encoded_data_path)
    return encoded_dataset


if __name__ == "__main__":
    encode_data(force=True, verbose=True, only_recent=True)
