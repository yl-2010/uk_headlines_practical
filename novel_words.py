"""Count novel words in UK test relative to UK train and AG train.

Writes artifacts/novel_words_summary.json (used by novel-words.html).
Tokenization: lowercase whitespace split — same word definition as lengths.html.

Per-label stats use UK test headlines of that label against the *full*
UK / AG training vocabularies (same novelty definition as the overall counts).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "novel_words_summary.json"
LABELS = ("business", "sport", "tech", "politics")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tokens(text: str) -> list[str]:
    return text.lower().split()


def vocab_from_texts(texts: list[str]) -> set[str]:
    v: set[str] = set()
    for t in texts:
        v.update(tokens(t))
    return v


def vocab_from_rows(rows: list[dict[str, str]]) -> set[str]:
    return vocab_from_texts([r["text"] for r in rows])


def top_wordlike(novel: set[str], counts: Counter[str], n: int = 20) -> list[str]:
    items = sorted(((w, counts[w]) for w in novel), key=lambda x: (-x[1], x[0]))
    return [
        w
        for w, _ in items
        if re.fullmatch(r"[a-z][a-z'-]*[a-z]|[a-z]", w)
    ][:n]


def novel_stats(
    test_rows: list[dict[str, str]],
    v_uk_train: set[str],
    v_ag_train: set[str],
) -> dict:
    texts = [r["text"] for r in test_rows]
    v_test = vocab_from_texts(texts)
    novel_vs_uk = v_test - v_uk_train
    novel_vs_ag = v_test - v_ag_train
    novel_vs_both = v_test - v_uk_train - v_ag_train

    counts: Counter[str] = Counter()
    tok_total = novel_uk_inst = novel_ag_inst = novel_both_inst = 0
    for t in texts:
        for w in tokens(t):
            counts[w] += 1
            tok_total += 1
            if w in novel_vs_uk:
                novel_uk_inst += 1
            if w in novel_vs_ag:
                novel_ag_inst += 1
            if w in novel_vs_both:
                novel_both_inst += 1

    vocab_n = len(v_test)
    return {
        "n": len(test_rows),
        "vocab": vocab_n,
        "novel_vs_uk_train": len(novel_vs_uk),
        "novel_vs_ag_train": len(novel_vs_ag),
        "novel_vs_both": len(novel_vs_both),
        "pct_vocab_novel_vs_uk": round(100 * len(novel_vs_uk) / vocab_n, 1) if vocab_n else 0.0,
        "pct_vocab_novel_vs_ag": round(100 * len(novel_vs_ag) / vocab_n, 1) if vocab_n else 0.0,
        "test_token_instances": tok_total,
        "novel_vs_uk_instances": novel_uk_inst,
        "novel_vs_ag_instances": novel_ag_inst,
        "novel_vs_both_instances": novel_both_inst,
        "pct_tokens_novel_vs_uk": round(100 * novel_uk_inst / tok_total, 1) if tok_total else 0.0,
        "pct_tokens_novel_vs_ag": round(100 * novel_ag_inst / tok_total, 1) if tok_total else 0.0,
        "examples_novel_vs_uk_wordlike": top_wordlike(novel_vs_uk, counts, n=12),
        "examples_novel_vs_ag_wordlike": top_wordlike(novel_vs_ag, counts, n=12),
    }


def main() -> None:
    uk_train = load_rows(DATA / "uk_train.csv")
    uk_test = load_rows(DATA / "uk_test.csv")
    ag_train = load_rows(DATA / "ag_train.csv")

    v_uk_train = vocab_from_rows(uk_train)
    v_ag_train = vocab_from_rows(ag_train)

    overall = novel_stats(uk_test, v_uk_train, v_ag_train)

    by_label_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in uk_test:
        by_label_rows[row["label"]].append(row)

    by_label = {
        label: novel_stats(by_label_rows[label], v_uk_train, v_ag_train)
        for label in LABELS
    }

    summary = {
        "tokenization": "lowercase whitespace split (same as lengths.html word counts)",
        "novelty_definition": (
            "word type in UK test (or UK test subset for a label) that never "
            "appears in the full UK / AG training vocabulary"
        ),
        "n_uk_train": len(uk_train),
        "n_uk_test": len(uk_test),
        "n_ag_train": len(ag_train),
        "vocab_uk_train": len(v_uk_train),
        "vocab_uk_test": overall["vocab"],
        "vocab_ag_train": len(v_ag_train),
        "novel_vs_uk_train": overall["novel_vs_uk_train"],
        "novel_vs_ag_train": overall["novel_vs_ag_train"],
        "novel_vs_both": overall["novel_vs_both"],
        "shared_with_uk_train": len(vocab_from_rows(uk_test) & v_uk_train),
        "shared_with_ag_train": len(vocab_from_rows(uk_test) & v_ag_train),
        "pct_test_vocab_novel_vs_uk": overall["pct_vocab_novel_vs_uk"],
        "pct_test_vocab_novel_vs_ag": overall["pct_vocab_novel_vs_ag"],
        "test_token_instances": overall["test_token_instances"],
        "novel_vs_uk_instances": overall["novel_vs_uk_instances"],
        "novel_vs_ag_instances": overall["novel_vs_ag_instances"],
        "novel_vs_both_instances": overall["novel_vs_both_instances"],
        "examples_novel_vs_uk_wordlike": overall["examples_novel_vs_uk_wordlike"],
        "examples_novel_vs_ag_wordlike": overall["examples_novel_vs_ag_wordlike"],
        "examples_novel_vs_both_wordlike": top_wordlike(
            vocab_from_rows(uk_test) - v_uk_train - v_ag_train,
            Counter(w for r in uk_test for w in tokens(r["text"])),
        ),
        "by_label": by_label,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        f"Novel vs UK train: {summary['novel_vs_uk_train']} "
        f"({summary['pct_test_vocab_novel_vs_uk']}% of UK test vocab)"
    )
    print(
        f"Novel vs AG train: {summary['novel_vs_ag_train']} "
        f"({summary['pct_test_vocab_novel_vs_ag']}% of UK test vocab)"
    )
    for label in LABELS:
        s = by_label[label]
        print(
            f"  {label}: novel_vs_uk={s['novel_vs_uk_train']} "
            f"({s['pct_vocab_novel_vs_uk']}%), "
            f"novel_vs_ag={s['novel_vs_ag_train']} "
            f"({s['pct_vocab_novel_vs_ag']}%)"
        )


if __name__ == "__main__":
    main()
