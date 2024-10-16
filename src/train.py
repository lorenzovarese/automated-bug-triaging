from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import datasets
import os, multiprocessing

MODEL_NAME = "bert-base-uncased"
CONTEXT_LENGTH = 512
NUM_PROC = min(100, multiprocessing.cpu_count() - 1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["cleaned_body"], padding="max_length", truncation=True, max_length=CONTEXT_LENGTH)

def encode_dataset(force=False, filename=os.path.join("data", "encoded_dataset")):
    if not force and os.path.exists(filename):
        return datasets.load_from_disk(filename)

    issues_df = pd.read_json(os.path.join("data", "issues.json"))
    issues_df["label"] = issues_df["assignee"].astype("category").cat.codes

    train_df = issues_df[issues_df["github_id"] <= 210_000]
    train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])
    test_df = issues_df[(210_000 < issues_df["github_id"]) & (issues_df["github_id"] <= 220_000)]

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "test": datasets.Dataset.from_pandas(test_df),
        "valid": datasets.Dataset.from_pandas(valid_df),
    })
    dataset = dataset.filter(lambda x: len(tokenizer(x["cleaned_body"])["input_ids"]) <= CONTEXT_LENGTH, num_proc=NUM_PROC)
    
    encoded_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[column for column in dataset['train'].column_names if column != "label"], 
        num_proc=NUM_PROC
    )
    encoded_dataset.save_to_disk(filename)
    return encoded_dataset

def main():
    encoded_dataset = encode_dataset(force=not True)
    encoded_dataset.set_format("torch")

    labels = set(encoded_dataset["train"]["label"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        save_strategy = "epoch",
        output_dir=os.path.join("data", "checkpoints"),
        overwrite_output_dir=False,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=5,
        fp16=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()

    

if __name__ == "__main__":
    main()
