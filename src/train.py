from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import datasets
import os, multiprocessing

TRAIN_MODEL = True
MODEL_NAME = "bert-base-uncased"
CONTEXT_LENGTH = 512
NUM_PROC = min(100, multiprocessing.cpu_count() - 1)
FRAC_OF_DATA = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_issues_df():
    issues_df = pd.read_json(os.path.join("data", "issues.json"))
    issues_df['label'] = issues_df['assignee'].astype('category').cat.codes
    if FRAC_OF_DATA < 1:
        issues_df = issues_df.sample(frac=FRAC_OF_DATA, random_state=1, weights="label").reset_index(drop=True)
    return issues_df

def tokenize(examples):
    encoding  = tokenizer(examples["cleaned_body"], padding="max_length", truncation=True, max_length=CONTEXT_LENGTH)
    encoding["label"] = examples["label"]
    return encoding

def encode_dataset(force=False, filename=os.path.join("data", f"encoded_dataset_{int(FRAC_OF_DATA*100):03d}")):
    if not force and os.path.exists(filename):
        return datasets.load_from_disk(filename)
    print(f"Saving encoded dataset to {filename}")

    issues_df = get_issues_df()

    train_df = issues_df[issues_df["github_id"] <= 210_000]
    train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])
    test_df = issues_df[(210_000 < issues_df["github_id"]) & (issues_df["github_id"] <= 220_000)]

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "eval": datasets.Dataset.from_pandas(eval_df),
        "test": datasets.Dataset.from_pandas(test_df),
    })
    # dataset = dataset.filter(lambda x: len(tokenizer(x["cleaned_body"])["input_ids"]) <= CONTEXT_LENGTH, num_proc=NUM_PROC)
    
    encoded_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[column for column in dataset['train'].column_names if column not in ["label", "github_id"]], 
        # remove_columns=dataset['train'].column_names,
        num_proc=NUM_PROC
    )
    encoded_dataset.save_to_disk(filename)
    return encoded_dataset

def train_model(model, dataset, output_dir=os.path.join("data", "checkpoints")):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        save_strategy = "epoch",
        output_dir=output_dir,
        overwrite_output_dir=False,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=5,
        fp16=True,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )

    print("----------- TRAINING -----------")
    print(f"Started at {pd.Timestamp.now()}")
    print(trainer.train())
    print(f"Ended at {pd.Timestamp.now()}")

    print("----------- Evaluation -----------")
    print(f"Started at {pd.Timestamp.now()}")
    print(trainer.evaluate())
    print(f"Ended at {pd.Timestamp.now()}")
    return trainer

def main():
    encoded_dataset = encode_dataset()
    encoded_dataset.set_format("torch")

    labels = set(map(lambda x: int(x), encoded_dataset["train"]["label"]))
    n_labels = max(labels) + 1 # can be different than len(labels) if sampled, because category 0 is removed for some reason (prolly too few issues assigned)

    issues_df = get_issues_df()
    label2assignee = issues_df[["label", "assignee"]].drop_duplicates().set_index("label")["assignee"].to_dict()

    ## Train the model
    if TRAIN_MODEL:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels)
        trainer = train_model(model, encoded_dataset)
        model = trainer.model
    else:
        model = AutoModelForSequenceClassification.from_pretrained("./data/checkpoints/checkpoint-14265")
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    accuracy = 0
    for example in encoded_dataset["test"]:
        encoding = {k: v.to("cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0) for k,v in example.items() if k not in ["github_id", "label"]}
        output = model(**encoding)
        logits = output.logits
        predicted_assignee = torch.argmax(logits, dim=1).item()
        expected_assignee = example["label"].item()
        accuracy += int(predicted_assignee == expected_assignee)

    print(f"Accuracy: {accuracy / len(encoded_dataset['test'])*100:.2f}%")

if __name__ == "__main__":
    main()
