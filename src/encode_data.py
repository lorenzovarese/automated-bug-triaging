from transformers import AutoTokenizer
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
import os, multiprocessing

MODEL_NAME = "bert-base-uncased"
CONTEXT_LENGTH = 512
NUM_PROC = min(100, multiprocessing.cpu_count() - 1)
FRAC_OF_DATA = 0.1

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    encoding  = TOKENIZER(examples["text"], padding="max_length", truncation=True, max_length=CONTEXT_LENGTH)
    encoding["label"] = examples["label"]
    return encoding

def train_eval_test_split(df, test_size=0.1):
    train_df = df[df["github_id"] <= 210_000]
    train_df, eval_df = train_test_split(train_df, test_size=test_size, random_state=42, stratify=train_df["label"])
    test_df = df[(210_000 < df["github_id"]) & (df["github_id"] <= 220_000)]

    return train_df, eval_df, test_df


def encode_data(
        encoded_data_path=os.path.join("data", f"encoded_data_{FRAC_OF_DATA*100:03.0f}"),
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

    issues_df["label"] = issues_df["assignee"].astype("category").cat.codes

    if FRAC_OF_DATA < 1:
        issues_df = issues_df.sample(frac=FRAC_OF_DATA, random_state=42, weights="label") # sample to reduce size (for testing purposes

    # filter issues that have more tokens that the model can handle
    if verbose: print(f"Filtering out issues with more than {CONTEXT_LENGTH} tokens...")

    # suppress annoying tokenization warnings
    loggers = {}
    for name in logging.root.manager.loggerDict:
        if "tokenization" in name:
            loggers[name] = logging.getLogger(name).getEffectiveLevel()
            logging.getLogger(name).setLevel(logging.ERROR)

    issues_df = issues_df[issues_df["text"].apply(lambda x: len(TOKENIZER(x)["input_ids"]) <= CONTEXT_LENGTH)]

    # restore logging levels
    for name, level in loggers.items():
        logging.getLogger(name).setLevel(level)


    # filter out issues with only one assignee because we can't split them
    assignees_counts = issues_df["assignee"].value_counts()

    if verbose: print(f"Filtering out {len(assignees_counts[assignees_counts <= 3])} assignees with less than 3 issues...")
    assignees_with_multiple_issues = assignees_counts[assignees_counts > 3].index
    issues_df = issues_df[issues_df["assignee"].isin(assignees_with_multiple_issues)]

    train_df, eval_df, test_df = train_eval_test_split(issues_df)

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
