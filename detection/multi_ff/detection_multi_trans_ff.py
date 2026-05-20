import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
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

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss

load_dotenv()
import random

# Hyperparametre


HP = {
    "test_nr": "mt base ff x",
    "max_length": 350,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "epochs": 5,
    "weight_decay": 0.05,
    "warmup_ratio": 0.05,
    "model_name": "xlm-roberta-base",  # roberta
    "random_state": 2018,
    "output_dir": "./detection_results/results",
    "logging_dir": "./detection_results/logs",
    "gamma": 0.5,
    "factor": 4,
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

# downsample the majority class (none) to balance the dataset
none_data = training_data[training_data["labels"] == 0]
other_data = training_data[training_data["labels"] != 0]
none_data_downsampled = none_data.sample(
    n=int(len(none_data) // HP["factor"]), random_state=HP["random_state"]
)
training_data = pd.concat([none_data_downsampled, other_data]).reset_index(drop=True)

test_data = pd.read_csv("../data/test_data.csv")

test_data["labels"] = test_data["multiclass"].map(label_mapping).fillna(0).astype(int)

tokenizer = AutoTokenizer.from_pretrained(HP["model_name"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataset = Dataset.from_pandas(training_data).shuffle(seed=HP["random_state"])
test_dataset = Dataset.from_pandas(test_data).shuffle(seed=HP["random_state"])


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

# training_args = TrainingArguments("test-trainer", eval_strategy="epoch")

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
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=50,
    logging_strategy="epoch",
    logging_dir=HP["logging_dir"],
    fp16=torch.cuda.is_available(),
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    report_to="wandb",
    run_name=f"run-{HP['test_nr']}",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)


model = AutoModelForSequenceClassification.from_pretrained(
    HP["model_name"], num_labels=5
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
roc_auc = evaluate.load("roc_auc")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]

    f1_macro = f1.compute(predictions=preds, references=labels, average="macro")["f1"]

    f1_weighted = f1.compute(predictions=preds, references=labels, average="weighted")[
        "f1"
    ]

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


classes = np.array([0, 1, 2, 3, 4])

class_weights = compute_class_weight(
    class_weight="balanced",
    # classes=np.unique(training_data["labels"]),
    classes=classes,
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
        # loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
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
trainer.evaluate()

predictions = trainer.predict(test_dataset_tokenized)

logits = predictions.predictions
labels = predictions.label_ids
y_true = labels


prob_matrix = torch.softmax(torch.tensor(logits), dim=1).numpy()
preds = np.argmax(prob_matrix, axis=1)

# --- classification report as wandb Table ---
class_names = [
    "none",
    "objectification",
    "stereotyping-dominance",
    "misogyny-violence",
    "ideological-inequality",
]

report = classification_report(
    y_true, preds, target_names=class_names, output_dict=True
)

columns = ["class", "precision", "recall", "f1-score", "support"]
table = wandb.Table(columns=columns)

for class_name in class_names:
    row = report[class_name]
    table.add_data(
        class_name, row["precision"], row["recall"], row["f1-score"], row["support"]
    )

for summary_key in ["macro avg", "weighted avg"]:
    row = report[summary_key]
    table.add_data(
        summary_key, row["precision"], row["recall"], row["f1-score"], row["support"]
    )

# --- scalar test F1s for cross-run comparison ---
test_f1_macro = report["macro avg"]["f1-score"]
test_f1_weighted = report["weighted avg"]["f1-score"]

inv_label_mapping = {v: k for k, v in label_mapping.items()}

test_df = pd.DataFrame({
    "id": test_dataset_tokenized["id"],
    "text": test_dataset_tokenized["text"],
    "true_label": y_true,
    "predicted": preds,
})
test_df = test_df.set_index("id")
test_df["true_label_name"] = test_df["true_label"].map(inv_label_mapping)
test_df["predicted_name"] = test_df["predicted"].map(inv_label_mapping)

misclassified = test_df[test_df["predicted"] != test_df["true_label"]].copy()
# For multiclass, confidence = probability of the predicted class
misclassified["confidence"] = prob_matrix[
    misclassified.index.map(lambda i: test_df.index.get_loc(i)), misclassified["predicted"]
]
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
        "classification_report": table,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        # ADDED: per-class F1 scalars
        **{
            f"test_f1_{class_name}": report[class_name]["f1-score"]
            for class_name in class_names
        },
        # these were already there
        "pr_curve": wandb.plot.pr_curve(y_true, prob_matrix),
        "roc_curve": wandb.plot.roc_curve(y_true, prob_matrix),
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=preds,
            class_names=class_names,
        ),
        #"misclassified_examples": error_table,
    }
)


wandb.finish()
