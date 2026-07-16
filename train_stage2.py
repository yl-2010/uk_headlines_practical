"""Stage 2: continue fine-tune Stage-1 DistilBERT on UK train; early-stop on UK val."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STAGE1 = ROOT / "artifacts" / "model_ag_headlines"
OUT_DIR = ROOT / "artifacts" / "model_uk_finetuned"
LABELS = ["business", "sport", "tech", "politics"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
MAX_LEN = 64


def get_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_frame(path: Path) -> Dataset:
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].map(LABEL2ID)
    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main() -> None:
    if not STAGE1.exists():
        raise FileNotFoundError(f"Missing Stage-1 checkpoint: {STAGE1}")

    device = get_device_str()
    print(f"device={device}; loading {STAGE1}")

    train_ds = load_frame(DATA_DIR / "uk_train.csv")
    val_ds = load_frame(DATA_DIR / "uk_val.csv")

    tokenizer = AutoTokenizer.from_pretrained(STAGE1)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(STAGE1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=42,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"Saved Stage-2 model to {OUT_DIR}")


if __name__ == "__main__":
    main()
