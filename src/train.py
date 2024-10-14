from transformers import AutoTokenizer
import pandas as pd
import datasets
import os


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def preprocess_data(examples):
    # take a batch of texts
    text = examples["cleaned_body"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add label
    encoding["labels"] = examples["assignee"]
    
    return encoding

def encode_dataset(dataset, force=False, filename="data/encoded_dataset"):
    if not force and os.path.exists(filename):
        return datasets.load_from_disk(filename)
    
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.save_to_disk(filename)
    return encoded_dataset


def main():
    train_df = pd.read_json("data/train/train_issues.json")
    test_df = pd.read_json("data/test/test_issues.json")

    issues_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    assignees = issues_df.assignee.unique()

    dataset = datasets.DatasetDict({"train": datasets.Dataset.from_pandas(train_df), "test": datasets.Dataset.from_pandas(test_df)})

    labels = assignees
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    encoded_dataset = encode_dataset(dataset)

    example = encoded_dataset['train'][0]
    print(example.keys())

    tokenizer.decode(example['input_ids'])

if __name__ == "__main__":
    main()
