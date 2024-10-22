from transformers import AutoTokenizer
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
import os, multiprocessing
import argparse

MODEL_NAME = "bert-base-uncased"
CONTEXT_LENGTH = 512
NUM_PROC = min(100, multiprocessing.cpu_count() - 1)

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
        encoded_data_path=os.path.join("data", f"encoded_data"),
        data_path=os.path.join("data", "issues.json"), 
        frac_of_data=1,
        only_recent=False,
        classical_preprocessing=False,
        num_proc=NUM_PROC,
        force=False,
        verbose=False,
    ):
    encoded_data_path = encoded_data_path + f"_{frac_of_data*100:03.0f}"

    if classical_preprocessing:
        encoded_data_path = encoded_data_path + "_classical"

    if only_recent:
        encoded_data_path = encoded_data_path + "_recent"

    if not force and os.path.exists(encoded_data_path):
        if verbose: print(f"Loading encoded data from '{encoded_data_path}'...")
        return datasets.load_from_disk(encoded_data_path)

    if verbose: 
        if force: print("Forcing re-encoding of data.")
        else: print(f"No cached data found at {encoded_data_path}.")
        print(f"Loading data from '{data_path}'...")

    issues_df = pd.read_json(data_path)

    if only_recent:
        if verbose: print("Filtering only recent data...")
        issues_df = issues_df[190_000 <= issues_df["github_id"]]

    issues_df["label"] = issues_df["assignee"].astype("category").cat.codes

    if frac_of_data < 1:
        issues_df = issues_df.sample(frac=frac_of_data, random_state=42, weights="label") # sample to reduce size (for testing purposes

    if classical_preprocessing:
        if verbose: print("Applying classical preprocessing...")
        issues_df["text"] = issues_df["classical_preprocessed_title"] + " " + issues_df["classical_preprocessed_body"]

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
        num_proc=num_proc,
    )

    if verbose: print(f"Saving encoded data to {encoded_data_path}...")
    encoded_dataset.save_to_disk(encoded_data_path)
    return encoded_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode the dataset for training a model.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-encoding of data.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("-r", "--only-recent", action="store_true", help="Only encode recent data.")
    parser.add_argument("-c", "--classical-preprocessing", action="store_true", help="Use classical preprocessing (stemming + stopwords removal) instead of the raw cleaned body of the issue.")
    parser.add_argument("--frac-of-data", type=float, default=1, help="Fraction of data to encode. Default is 1. Use a smaller value (between 0 and 1) for testing.")
    parser.add_argument("--data-path", type=str, default=os.path.join("data", "issues.json"), help="Path to the dataset. Default is 'data/issues.json'.")
    parser.add_argument("--encoded-data-path", type=str, default=os.path.join("data", "encoded_data"), help="Path to save the encoded dataset. Default is 'data/encoded_data'. Note: The path is then extended with the fraction of the data, together with whether it is only recent issues or not.")
    parser.add_argument("--num-proc", type=int, default=NUM_PROC, help=f"Number of processes to use for encoding. Default is {NUM_PROC}.")

    
    args = parser.parse_args()
    encode_data(
        encoded_data_path=args.encoded_data_path,
        data_path=args.data_path,
        frac_of_data=args.frac_of_data,
        only_recent=args.only_recent,
        classical_preprocessing=args.classical_preprocessing,
        force=args.force,
        verbose=args.verbose,
    )
