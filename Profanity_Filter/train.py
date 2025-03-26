import pandas as pd
import matplotlib.pyplot as plt
import torch
from model.data_loader import main
from evaluate import compute_metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 상수 정의
NUM_LABELS = 3
OUTPUT_DIR = './output/'
NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 64
LOGGING_DIR = './output/logs'
LOGGING_STEPS = 500
SAVE_TOTAL_LIMIT = 2

def setup_device():
    """GPU 설정."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    return device

def load_data_and_model(device):
    """데이터셋과 모델 로드."""
    train_dataset, valid_dataset, MODEL_NAME = main()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)
    return train_dataset, valid_dataset, model

def train_model(train_dataset, valid_dataset, model):
    """모델 학습."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(eval_dataset=valid_dataset)

if __name__ == "__main__":
    device = setup_device()
    train_dataset, valid_dataset, model = load_data_and_model(device)
    train_model(train_dataset, valid_dataset, model)
