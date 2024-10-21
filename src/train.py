from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import os, multiprocessing
from encode_data import encode_data, TOKENIZER
import evaluate

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
    encoded_dataset = encode_data(only_recent=True)
    encoded_dataset.set_format("torch")

    labels = set(map(lambda x: int(x), encoded_dataset["train"]["label"]))
    n_labels = max(labels) + 1 # can be different than len(labels) if sampled, because category 0 is removed for some reason (prolly too few issues assigned)

    ## Train the model
    if TRAIN_MODEL:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("./data/checkpoints/checkpoint-2095")

    trainer = trainer_for_model(model, encoded_dataset)
    if TRAIN_MODEL:
        trainer.train()

    eval_acc = trainer.evaluate()["eval_accuracy"]
    print(f"Evaluation: {eval_acc*100:.2f}%")

    test_acc = trainer.evaluate(encoded_dataset["test"])["eval_accuracy"]
    print(f"Test: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
