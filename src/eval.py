from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import os
from encode_data import encode_data, TOKENIZER
import evaluate
import argparse

MODEL_NAME = "bert-base-uncased"

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-model", action="store_true", help="Force the training of the model.")
    parser.add_argument("--frac-of-data", type=float, default=1, help="Fraction of data to use for training. Default is 1. Use a smaller value (between 0 and 1) for testing.")
    parser.add_argument("-r", "--only-recent", action="store_true", help="Use only recent data for training.")
    parser.add_argument("-c", "--checkpoint", type=str, default="", help="Path to a checkpoint to load. Ignored if --train-model is used.")
    parser.add_argument("--classical-preprocessing", action="store_true", help="Use classical preprocessing (stemming + stopwords removal) instead of the raw cleaned body of the issue.")
    args = parser.parse_args()

    encoded_dataset = encode_data(
        only_recent=args.only_recent, 
        frac_of_data=args.frac_of_data,
        classical_preprocessing=args.classical_preprocessing,
    )
    encoded_dataset.set_format("torch")

    labels = set(map(lambda x: int(x), encoded_dataset["train"]["label"]))
    n_labels = max(labels) + 1 # can be different than len(labels) if sampled, because category 0 is removed for some reason (prolly too few issues assigned)

    ## Train the model
    if args.train_model:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels)
    else:
        if args.checkpoint:
            model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
        else:
            raise ValueError("You need to provide a checkpoint path if you're not forcing training")

    trainer = trainer_for_model(model, encoded_dataset)
    if args.train_model:
        trainer.train()

    eval_acc = trainer.evaluate()["eval_accuracy"]
    test_acc = trainer.evaluate(encoded_dataset["test"])["eval_accuracy"]

    print(f"Accuracy on evaluation set: {eval_acc*100:.2f}%")
    print(f"Accuracy on test set: {test_acc*100:.2f}%")
