"""Prepare AG News titles + BBC UK headlines with unified 4-way labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
AG_TRAIN_CAP = 30_000
DATA_DIR = Path(__file__).resolve().parent / "data"

LABELS = ["business", "sport", "tech", "politics"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# AG News: 0 World, 1 Sports, 2 Business, 3 Sci/Tech
AG_LABEL_MAP = {
    0: "politics",  # World ↔ politics (imperfect, intentional)
    1: "sport",
    2: "business",
    3: "tech",
}

BBC_LABEL_MAP = {
    "business": "business",
    "sport": "sport",
    "tech": "tech",
    "politics": "politics",
    # entertainment dropped
}


def first_sentence_or_tokens(text: str, max_tokens: int = 16) -> str:
    """Headline proxy: first sentence, else first ~16 whitespace tokens."""
    text = (text or "").strip()
    if not text:
        return ""
    # Prefer a short first sentence.
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    candidate = parts[0].strip()
    tokens = candidate.split()
    if len(tokens) <= max_tokens:
        return candidate
    return " ".join(text.split()[:max_tokens])


def ag_title_from_text(raw: str) -> str:
    """fancyzhx/ag_news has no title column; text is usually 'Title - body'."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if " - " in raw:
        return raw.split(" - ", 1)[0].strip()
    # Fallback: short headline proxy (first sentence / ~16 tokens)
    return first_sentence_or_tokens(raw, max_tokens=16)


def prepare_ag() -> pd.DataFrame:
    ds = load_dataset("fancyzhx/ag_news", split="train")
    rows = []
    for ex in ds:
        title = (ex.get("title") or "").strip() or ag_title_from_text(ex.get("text") or "")
        if not title:
            continue
        label = AG_LABEL_MAP[int(ex["label"])]
        rows.append({"text": title, "label": label})
    df = pd.DataFrame(rows)
    # Stratified subsample for Stage-1 speed.
    if len(df) > AG_TRAIN_CAP:
        df, _ = train_test_split(
            df,
            train_size=AG_TRAIN_CAP,
            stratify=df["label"],
            random_state=SEED,
        )
    return df.reset_index(drop=True)


def prepare_bbc() -> pd.DataFrame:
    ds = load_dataset("SetFit/bbc-news", split="train")
    # Also pull test if present and concatenate before our stratified split.
    try:
        ds_test = load_dataset("SetFit/bbc-news", split="test")
        from datasets import concatenate_datasets

        ds = concatenate_datasets([ds, ds_test])
    except Exception:
        pass

    rows = []
    for ex in ds:
        raw_label = (ex.get("label_text") or ex.get("label") or "").strip().lower()
        if isinstance(ex.get("label"), int) and "label_text" not in ex:
            # Fallback if only int labels; SetFit usually has label_text.
            continue
        if raw_label == "entertainment":
            continue
        if raw_label not in BBC_LABEL_MAP:
            # Sometimes label is int and label_text present separately
            raw_label = str(ex.get("label_text", "")).strip().lower()
            if raw_label == "entertainment" or raw_label not in BBC_LABEL_MAP:
                continue
        label = BBC_LABEL_MAP[raw_label]

        # Prefer a real title field if present.
        if ex.get("title"):
            text = str(ex["title"]).strip()
        elif ex.get("headline"):
            text = str(ex["headline"]).strip()
        else:
            body = str(ex.get("text") or "").strip()
            text = first_sentence_or_tokens(body)

        if not text:
            continue
        rows.append({"text": text, "label": label})

    return pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)


def split_uk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AG News…")
    ag = prepare_ag()
    ag_train, ag_val = train_test_split(
        ag,
        test_size=0.05,
        stratify=ag["label"],
        random_state=SEED,
    )
    ag_train.to_csv(DATA_DIR / "ag_train.csv", index=False)
    ag_val.to_csv(DATA_DIR / "ag_val.csv", index=False)
    print(f"  AG train={len(ag_train)} val={len(ag_val)}")
    print(ag_train["label"].value_counts().to_string())

    print("Loading BBC UK…")
    uk = prepare_bbc()
    print(f"  BBC after drop entertainment / dedupe: {len(uk)}")
    print(uk["label"].value_counts().to_string())

    uk_train, uk_val, uk_test = split_uk(uk)
    uk_train.to_csv(DATA_DIR / "uk_train.csv", index=False)
    uk_val.to_csv(DATA_DIR / "uk_val.csv", index=False)
    uk_test.to_csv(DATA_DIR / "uk_test.csv", index=False)
    print(
        f"  UK split train={len(uk_train)} val={len(uk_val)} test={len(uk_test)} "
        f"(seed={SEED}, frozen test)"
    )
    print("Done. Wrote CSVs under data/")


if __name__ == "__main__":
    main()
