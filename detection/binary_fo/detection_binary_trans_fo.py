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
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve
from torch.nn import CrossEntropyLoss

import evaluate
from datasets import Dataset, DatasetDict

from dotenv import load_dotenv
import os
import random

from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn

load_dotenv()


# Hyperparametre


HP = {
    "test_nr": "bt base focal detERRTrans",
    "max_length": 350,
    "batch_size": 16,
    "learning_rate": 4e-6,
    "epochs": 10,
    "weight_decay": 0.1,
    "warmup_ratio": 0.05,
    "model_name": "xlm-roberta-base",
    "random_state": 2018,
    "output_dir": "./detection_results/results",
    "logging_dir": "./detection_results/logs",
    "gamma": 1.0,
    "factor": 1.0,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Force deterministic CUDA ops — may slow training slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8: covers ops not controlled by the above
    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Required when using torch.use_deterministic_algorithms(True) with CUDA
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


set_seed(HP["random_state"])


wandb.login(key=os.getenv("WANDB_API_KEY"))

wandb.init(
    entity="sofiefe-ntnu",
    project="master_detection",
    name=f"run-{HP['test_nr']}",
    config=HP,
)

# region datasets
# import dataset

label_mapping = {"sexist": 1, "non-sexist": 0}

training_data = pd.read_csv("../data/training_data.csv")
training_data["labels"] = training_data["binary"].map(label_mapping)

# downsample the majority class (none) to balance the dataset
none_data = training_data[training_data["labels"] == 0]
other_data = training_data[training_data["labels"] != 0]
none_data_downsampled = none_data.sample(
    n=int(len(none_data) // HP["factor"]), random_state=HP["random_state"]
)
training_data = pd.concat([none_data_downsampled, other_data]).reset_index(drop=True)

test_data = pd.read_csv("../data/test_data.csv")
test_data["labels"] = test_data["binary"].map(label_mapping)

tokenizer = AutoTokenizer.from_pretrained(HP["model_name"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = Dataset.from_pandas(training_data).shuffle(seed=HP["random_state"])
test_dataset = Dataset.from_pandas(test_data)


def tokenize_function_trans(example):
    texts = [t if isinstance(t, str) else "" for t in example["translated_text"]]
    return tokenizer(texts, truncation=True, max_length=HP["max_length"])


def tokenize_function(example):
    return tokenizer(
        example["text"], truncation=True, max_length=HP["max_length"]
    )


train_dataset = train_dataset.remove_columns(
    [
        "original_id",
        "id",
        "source_dataset",
        "lang",
        "text",
        "binary",
        "multiclass",
    ]
)
train_dataset_tokenized = train_dataset.map(tokenize_function_trans, batched=True)

train_dataset_tokenized = train_dataset_tokenized.class_encode_column("labels")

train_valid_data = train_dataset_tokenized.train_test_split(
    test_size=0.1, stratify_by_column="labels", seed=HP["random_state"]
)

train_valid_data_dict = DatasetDict(
    {"train": train_valid_data["train"], "validation": train_valid_data["test"]}
)

test_dataset = test_dataset.remove_columns(["binary", "multiclass"])
test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)
# endregion


training_args = TrainingArguments(
    output_dir=HP["output_dir"] + "/checkpoints",
    seed=HP["random_state"],
    data_seed=HP["random_state"],  # controls DataLoader shuffling
    dataloader_num_workers=0,  # eliminates worker ordering randomness
    full_determinism=True,
    learning_rate=HP["learning_rate"],
    per_device_train_batch_size=HP["batch_size"],
    per_device_eval_batch_size=HP["batch_size"],
    num_train_epochs=HP["epochs"],
    weight_decay=HP["weight_decay"],
    warmup_ratio=HP["warmup_ratio"],
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    gradient_accumulation_steps=4,
    logging_strategy="epoch",
    logging_dir=HP["logging_dir"],
    fp16=torch.cuda.is_available(),
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    report_to="wandb",
    run_name=f"run-{HP['test_nr']}",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)


model = AutoModelForSequenceClassification.from_pretrained(
    HP["model_name"], num_labels=2
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
roc_auc = evaluate.load("roc_auc")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=preds, references=labels)["f1"]
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    auc_score = roc_auc.compute(prediction_scores=probs, references=labels)["roc_auc"]
    f2 = fbeta_score(labels, preds, beta=2)
    return {"accuracy": acc, "f1": f1_score, "auc": auc_score, "f2": f2}


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(training_data["labels"]),
    y=training_data["labels"],
)

class_weights = np.array(class_weights)
class_weights[0] = HP["factor"] * class_weights[0]

class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")


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


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = FocalLoss(weight=class_weights, gamma=HP["gamma"])
        # loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
    model,
    training_args,
    train_dataset=train_valid_data_dict["train"],
    eval_dataset=train_valid_data_dict["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


def best_threshold_from(pred_output):
    logits = pred_output.predictions
    labels = pred_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    return thresholds[np.argmax(f1s)], probs


val_preds = trainer.predict(train_valid_data_dict["validation"])
threshold, _ = best_threshold_from(val_preds)

test_preds = trainer.predict(test_dataset_tokenized)

_, test_probs = best_threshold_from(test_preds)
test_labels = test_preds.label_ids
prob_matrix = torch.softmax(torch.tensor(test_preds.predictions), dim=1).numpy()
final_preds = (test_probs > threshold).astype(int)

precision, recall, _ = precision_recall_curve(test_labels, test_probs)
f1s = 2 * precision * recall / (precision + recall + 1e-8)

honest_f1 = f1_score(test_labels, final_preds)
honest_f2 = fbeta_score(test_labels, final_preds, beta=2)

inv_label_mapping = {v: k for k, v in label_mapping.items()}

test_df = pd.DataFrame({
    "id": test_dataset_tokenized["id"],  # now preserved
    "text": test_dataset_tokenized["text"],
    "true_label": test_labels,
    "predicted": final_preds,
})
test_df = test_df.set_index("id")  # makes it easy to look up by original ID
test_df["predicted_name"] = test_df["predicted"].map(inv_label_mapping)
test_df["true_label_name"] = test_df["true_label"].map(inv_label_mapping)

misclassified = test_df[test_df["predicted"] != test_df["true_label"]].copy()
misclassified["confidence"] = np.where(
    misclassified["predicted"] == 1,
    test_probs[misclassified.index],
    1 - test_probs[misclassified.index],
)
misclassified = misclassified.sort_values("confidence", ascending=False)

error_table = wandb.Table(
    columns=["id", "text", "true_label", "predicted", "confidence"],
    data=[
        [idx, row["text"], row["true_label_name"], row["predicted_name"], row["confidence"]]
        for idx, row in misclassified.iterrows()
    ],
)


wandb.log(
    {
        "best_threshold": threshold,
        "best_f1": honest_f1,
        "best_f2": honest_f2,
        "pr_curve": wandb.plot.pr_curve(test_labels, prob_matrix),
        "roc_curve": wandb.plot.roc_curve(test_labels, prob_matrix),
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=test_labels, preds=final_preds, class_names=["non-sexist", "sexist"]
        ),
        #"misclassified_examples": error_table,
    }
)

wandb.finish()
