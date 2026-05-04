import os

import mlflow
import torch
from model.data_loader import main
from evaluate import compute_metrics
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

NUM_LABELS = 3
OUTPUT_DIR = './output/'
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, 'final')
NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 64
LOGGING_DIR = './output/logs'
LOGGING_STEPS = 500
SAVE_TOTAL_LIMIT = 2

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns')
MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT', 'profanity_filter')


def setup_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    return device


def load_data_and_model(device):
    train_dataset, valid_dataset, MODEL_NAME = main()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)
    return train_dataset, valid_dataset, model, MODEL_NAME


def build_trainer(train_dataset, valid_dataset, model):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to=["mlflow"],
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    device = setup_device()
    train_dataset, valid_dataset, model, base_model_name = load_data_and_model(device)
    trainer = build_trainer(train_dataset, valid_dataset, model)

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "framework": "transformers",
            "task": "profanity_classification",
            "base_model": base_model_name,
            "num_labels": NUM_LABELS,
        })
        trainer.train()
        trainer.evaluate(eval_dataset=valid_dataset)
        trainer.save_model(FINAL_MODEL_DIR)
        mlflow.log_artifacts(FINAL_MODEL_DIR, artifact_path="model")
        print(f"MLflow run: {run.info.run_id}")
