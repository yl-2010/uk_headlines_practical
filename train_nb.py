"""Multinomial Naive Bayes baselines: train on AG and on UK, score UK test."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
LABELS = ["business", "sport", "tech", "politics"]


def train_and_eval(train_csv: Path, test_csv: Path, model_path: Path, tag: str) -> dict:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    pipe = Pipeline(
        [
            ("bow", CountVectorizer(ngram_range=(1, 1), min_df=2)),
            ("nb", MultinomialNB()),
        ]
    )
    pipe.fit(train["text"].astype(str), train["label"])
    preds = pipe.predict(test["text"].astype(str))
    y_true = test["label"]

    metrics = {
        "system": tag,
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", labels=LABELS)),
        "micro_f1": float(f1_score(y_true, preds, average="micro", labels=LABELS)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_ag = train_and_eval(
        DATA_DIR / "ag_train.csv",
        DATA_DIR / "uk_test.csv",
        ARTIFACTS / "nb_ag.joblib",
        "NB @ AG",
    )
    nb_uk = train_and_eval(
        DATA_DIR / "uk_train.csv",
        DATA_DIR / "uk_test.csv",
        ARTIFACTS / "nb_uk.joblib",
        "NB @ UK",
    )
    out = {"nb_ag": nb_ag, "nb_uk": nb_uk}
    (ARTIFACTS / "metrics_nb.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {ARTIFACTS / 'metrics_nb.json'}")


if __name__ == "__main__":
    main()
