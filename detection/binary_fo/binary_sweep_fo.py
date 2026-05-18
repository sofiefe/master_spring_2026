import pandas as pd
import numpy as np

import torch
import wandb
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
import os

from torch.nn import CrossEntropyLoss

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

os.environ["WANDB_PROJECT"] = "master_detection"
os.environ["WANDB_ENTITY"] = "sofiefe-ntnu"


from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
import random

# Hyperparametre


HP = {
    "test_nr": "13 binary",
    "max_length": 350,
    "batch_size": 64,
    "learning_rate": 2e-5,
    "epochs": 10,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "model_name": "xlm-roberta-large",  # roberta
    "random_state": 2018,
    "output_dir": "./detection_results/results",
    "logging_dir": "./detection_results/logs",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# region datasets


label_mapping = {"sexist": 1, "non-sexist": 0}

training_data = pd.read_csv("../data/training_data.csv")
training_data["labels"] = training_data["binary"].map(label_mapping)

none_data = training_data[training_data["labels"] == 0]
other_data = training_data[training_data["labels"] != 0]

test_data = pd.read_csv("../data/test_data.csv")
test_data["labels"] = test_data["binary"].map(label_mapping)

tokenizer = AutoTokenizer.from_pretrained(HP["model_name"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = Dataset.from_pandas(training_data).shuffle(seed=HP["random_state"])
test_dataset = Dataset.from_pandas(test_data).shuffle(seed=HP["random_state"])


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=HP["max_length"])


train_dataset = train_dataset.remove_columns(
    [
        "original_id",
        "id",
        "source_dataset",
        "lang",
        "translated_text",
        "binary",
        "multiclass",
    ]
)
train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)

train_dataset_tokenized = train_dataset_tokenized.class_encode_column("labels")

train_valid_data = train_dataset_tokenized.train_test_split(
    test_size=0.1, stratify_by_column="labels", seed=HP["random_state"]
)

train_valid_data = DatasetDict(
    {"train": train_valid_data["train"], "validation": train_valid_data["test"]}
)

test_dataset = test_dataset.remove_columns(["id", "binary", "multiclass"])
test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight  # your existing class_weights
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# Metrics
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    roc_auc = evaluate.load("roc_auc")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=preds, references=labels)["f1"]
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    auc_score = roc_auc.compute(prediction_scores=probs, references=labels)["roc_auc"]
    return {"accuracy": acc, "f1": f1_score, "auc": auc_score}


# Model init function (fresh model for every run)
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        HP["model_name"], num_labels=2
    )


sweep_config = {
    "name": "b large factor + focal 2",  # BYTT NAVN MODEL
    "method": "bayes",
    "metric": {"name": "eval_f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4,
        },
        "batch_size": {"values": [16, 32, 64, 128]},
        "epochs": {"values": [3, 5, 10]},
        "weight_decay": {"values": [0.0, 0.01, 0.05, 0.1]},
        "warmup_ratio": {"values": [0.0, 0.05, 0.1]},
        "gamma": {"values": [0.5, 1.0, 1.5, 2.0]},
        #"factor": {"distribution": "int_uniform", "min": 1, "max": 25},
        "factor": {"values": [1]},
    },
}


def train():
    set_seed(HP["random_state"])
    run = wandb.init()
    config = wandb.config
    # Downsample none-class using sweep's factor
    none_downsampled = none_data.sample(
        n=len(none_data) // int(config.factor), random_state=HP["random_state"]
    )
    training_data_run = pd.concat([none_downsampled, other_data]).reset_index(drop=True)

    # Recompute class weights for this run
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=training_data_run["labels"],
    )
    cw = np.array(cw)
    cw[0] = config.factor * cw[0]
    class_weights_run = torch.tensor(cw, dtype=torch.float)

    # Rebuild the dataset for this run
    train_dataset_run = Dataset.from_pandas(training_data_run).shuffle(
        seed=HP["random_state"]
    )
    train_dataset_run = train_dataset_run.remove_columns(
        [
            "original_id",
            "id",
            "source_dataset",
            "lang",
            "translated_text",
            "binary",
            "multiclass",
        ]
    )
    train_dataset_run = train_dataset_run.map(tokenize_function, batched=True)
    train_dataset_run = train_dataset_run.class_encode_column("labels")

    train_valid_run = train_dataset_run.train_test_split(
        test_size=0.1, stratify_by_column="labels", seed=HP["random_state"]
    )
    train_valid_run = DatasetDict(
        {
            "train": train_valid_run["train"],
            "validation": train_valid_run["test"],
        }
    )

    # WeightedTrainer closes over class_weights_run
    class WeightedTrainer(Trainer):
        def __init__(self, *args, gamma=2.0, **kwargs):
            self.gamma = gamma
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = FocalLoss(
                weight=class_weights_run.to(logits.device), gamma=self.gamma
            )
            # loss_fct = CrossEntropyLoss(weight=class_weights_run.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    run_output_dir = f"./results/{run.name or run.id}"

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        seed=HP["random_state"],
        data_seed=HP["random_state"],
        dataloader_num_workers=0,
        full_determinism=True,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        report_to="wandb",
        fp16=torch.cuda.is_available(),
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = WeightedTrainer(
        gamma=config.gamma,
        model_init=model_init,
        args=training_args,
        train_dataset=train_valid_run["train"],
        eval_dataset=train_valid_run["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    wandb.log(final_metrics)

    run.finish()


sweep_id = wandb.sweep(
    sweep_config,
    project="master_detection",
    entity="sofiefe-ntnu",
)

wandb.agent(sweep_id, function=train, count=10)
