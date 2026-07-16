"""Evaluate a DistilBERT checkpoint on the frozen UK test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["business", "sport", "tech", "politics"]
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 32):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return np.array(preds)


def save_confusion(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confusion-name", default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"device={device}")

    test = pd.read_csv(DATA_DIR / "uk_test.csv")
    label2id = {n: i for i, n in enumerate(LABELS)}
    y_true = test["label"].map(label2id).to_numpy()
    texts = test["text"].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.to(device)

    y_pred = predict(model, tok, texts, device)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class = {name: float(report[name]["f1-score"]) for name in LABELS}

    payload = {
        "checkpoint": str(args.checkpoint),
        "n_test": int(len(test)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_f1": per_class,
        "device": str(device),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    conf_name = args.confusion_name
    if conf_name is None:
        stem = Path(args.checkpoint).name
        conf_name = f"confusion_{stem}.png"
    save_confusion(y_true, y_pred, ARTIFACTS / conf_name, title=Path(args.checkpoint).name)


if __name__ == "__main__":
    main()
