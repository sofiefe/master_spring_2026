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
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
import os

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
import random

# Hyperparametre

HP = {
    "test_nr": "multi ff sweep",
    "model_name": "xlm-roberta-base",  # roberta
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

label_mapping = {
    "none": 0,
    "objectification": 1,
    "stereotyping-dominance": 2,
    "misogyny-violence": 3,
    "ideological-inequality": 4,
}

training_data = pd.read_csv("../data/training_data.csv")
training_data["labels"] = (
    training_data["multiclass"].map(label_mapping).fillna(0).astype(int)
)

test_data = pd.read_csv("../data/test_data.csv")
test_data["labels"] = test_data["multiclass"].map(label_mapping).fillna(0).astype(int)

none_data = training_data[training_data["labels"] == 0]
other_data = training_data[training_data["labels"] != 0]

tokenizer = AutoTokenizer.from_pretrained(HP["model_name"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_dataset = Dataset.from_pandas(test_data)


def tokenize_function(example):
    return tokenizer(
        example["text"], truncation=True, padding=True, max_length=HP["max_length"]
    )


test_dataset = test_dataset.remove_columns(["id", "binary", "multiclass"])
test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)
# endregion


# region Focal Loss (from binary script)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# endregion


# region Metrics
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    f1_weighted = f1.compute(predictions=preds, references=labels, average="weighted")[
        "f1"
    ]

    return {
        "accuracy": acc,
        "f1": f1_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# endregion


sweep_config = {
    "name": "b large factor",  # BYTT NAVN MODEL
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
        "factor": {"distribution": "int_uniform", "min": 1, "max": 25},
    },
}


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        HP["model_name"], num_labels=5
    )


def train():
    set_seed(HP["random_state"])
    run = wandb.init()
    config = wandb.config

    run_output_dir = f"./results/{run.name or run.id}"

    # Downsample
    none_downsampled = none_data.sample(
        n=len(none_data) // int(config.factor), random_state=HP["random_state"]
    )
    training_data_run = pd.concat([none_downsampled, other_data]).reset_index(drop=True)

    # class weights
    classes = np.array([0, 1, 2, 3, 4])
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=training_data_run["labels"],
    )
    cw = np.array(cw)

    # Upweight the none class by factor (mirrors binary script)
    cw[0] = config.factor * cw[0]
    class_weights_run = torch.tensor(cw, dtype=torch.float)

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

    class WeightedTrainer(Trainer):
        def __init__(self, *args, gamma=2.0, **kwargs):
            self.gamma = gamma
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = FocalLoss(
                weight=class_weights_run.to(logits.device),
                gamma=self.gamma,
            )
            # loss_fct = CrossEntropyLoss(
            #     weight=class_weights_run.to(logits.device)
            # )
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

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
        per_device_eval_batch_size=config.batch_size,
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
