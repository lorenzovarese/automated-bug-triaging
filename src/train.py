from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd
import os, multiprocessing
from encode_data import encode_data, TOKENIZER

TRAIN_MODEL = True
MODEL_NAME = "bert-base-uncased"
NUM_PROC = min(100, multiprocessing.cpu_count() - 1)

accuracy = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def trainer_for_model(model, dataset, output_dir=os.path.join("data", "checkpoints")):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        save_strategy="epoch",
        eval_strategy="epoch",
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
        tokenizer=TOKENIZER,
        compute_metrics=compute_metrics,
    )

    return trainer

def main():
    encoded_dataset = encode_data()
    encoded_dataset.set_format("torch")

    labels = set(map(lambda x: int(x), encoded_dataset["train"]["label"]))
    n_labels = max(labels) + 1 # can be different than len(labels) if sampled, because category 0 is removed for some reason (prolly too few issues assigned)

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
