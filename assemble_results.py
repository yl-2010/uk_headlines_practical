"""Assemble results_table.md and a short error_analysis.md from metric JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
LABELS = ["business", "sport", "tech", "politics"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@torch.no_grad()
def collect_errors(checkpoint: Path, limit: int = 12) -> list[dict]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test = pd.read_csv(DATA_DIR / "uk_test.csv")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    rows = []
    for _, row in test.iterrows():
        text = str(row["text"])
        true = row["label"]
        enc = tok(text, truncation=True, max_length=64, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        pred_id = int(model(**enc).logits.argmax(dim=-1).item())
        pred = LABELS[pred_id]
        if pred != true:
            rows.append({"text": text, "true": true, "pred": pred})
        if len(rows) >= limit * 3:
            # gather extra then pick diverse later
            pass
    return rows[:limit]


def main() -> None:
    nb = load_json(ARTIFACTS / "metrics_nb.json")
    before = load_json(ARTIFACTS / "metrics_before.json")
    uk_only = load_json(ARTIFACTS / "metrics_uk_only.json")
    after = load_json(ARTIFACTS / "metrics_after.json")

    def micro(m: dict) -> float:
        # Fall back to accuracy for older metric JSONs (single-label micro-F1 ≈ acc).
        return float(m.get("micro_f1", m["accuracy"]))

    rows = [
        ("NB @ AG", nb["nb_ag"]["accuracy"], nb["nb_ag"]["macro_f1"], micro(nb["nb_ag"])),
        (
            "DistilBERT @ AG (before)",
            before["accuracy"],
            before["macro_f1"],
            micro(before),
        ),
        ("NB @ UK", nb["nb_uk"]["accuracy"], nb["nb_uk"]["macro_f1"], micro(nb["nb_uk"])),
        (
            "DistilBERT UK only (no AG)",
            uk_only["accuracy"],
            uk_only["macro_f1"],
            micro(uk_only),
        ),
        (
            "DistilBERT UK fine-tuned (after)",
            after["accuracy"],
            after["macro_f1"],
            micro(after),
        ),
    ]

    lines = [
        "# UK headlines results (frozen UK test)",
        "",
        "| System | UK test accuracy | UK test macro-F1 | UK test micro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, acc, macro, mic in rows:
        lines.append(f"| {name} | {acc:.4f} | {macro:.4f} | {mic:.4f} |")

    delta_macro = after["macro_f1"] - before["macro_f1"]
    delta_micro = micro(after) - micro(before)
    uk_vs_ag_macro = after["macro_f1"] - uk_only["macro_f1"]
    uk_vs_ag_micro = micro(after) - micro(uk_only)
    lines.extend(
        [
            "",
            f"Macro-F1 delta (AG→UK after − AG before): **{delta_macro:+.4f}**.",
            f"Micro-F1 delta (AG→UK after − AG before): **{delta_micro:+.4f}**.",
            f"Macro-F1 delta (AG→UK after − UK only): **{uk_vs_ag_macro:+.4f}**.",
            f"Micro-F1 delta (AG→UK after − UK only): **{uk_vs_ag_micro:+.4f}**.",
            "",
            "Notes:",
            "- Same frozen UK test set (seed 42, stratified 70/15/15) for all systems.",
            "- UK only = DistilBERT fine-tuned from base on UK train (no AG Stage 1).",
            "- AG→UK after = continued fine-tune from the AG Stage-1 checkpoint.",
            "- AG `World` was mapped to `politics` (imperfect but intentional).",
            "- BBC `entertainment` rows were dropped.",
            "- For single-label multi-class, micro-F1 often equals accuracy; both are reported for lect2 comparison.",
        ]
    )
    (ARTIFACTS / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Error analysis: examples Stage-1 misses that Stage-2 may fix.
    before_errs = collect_errors(ARTIFACTS / "model_ag_headlines", limit=20)
    after_ckpt = ARTIFACTS / "model_uk_finetuned"
    tok = AutoTokenizer.from_pretrained(after_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(after_ckpt)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    fixed = []
    still_wrong = []
    with torch.no_grad():
        for err in before_errs:
            enc = tok(err["text"], truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            pred = LABELS[int(model(**enc).logits.argmax(dim=-1).item())]
            item = {**err, "after_pred": pred}
            if pred == err["true"]:
                fixed.append(item)
            else:
                still_wrong.append(item)

    ea = [
        "# Error analysis (UK test)",
        "",
        "## Setup risks",
        "- AG `World` ↔ BBC `politics` is an imperfect label bridge.",
        "- BBC items are often article bodies; we use title if present, else first sentence / first ~16 tokens as a headline proxy.",
        "- UK data is small; Stage-2 uses early stopping on UK val only (test stays frozen).",
        "",
        "## What the before model tends to miss",
        "Stage-1 is trained on US-centric AG News titles. On UK BBC headlines it often misses:",
        "- British politics entities (Westminster, party names, ministers)",
        "- UK sport (Premier League clubs, cricket, rugby)",
        "- UK spellings / local brands in tech/business copy",
        "",
        f"Of {len(before_errs)} sampled Stage-1 errors, "
        f"{len(fixed)} were corrected after UK fine-tuning "
        f"and {len(still_wrong)} remained wrong in this sample.",
        "",
        "### Corrected after UK fine-tune (examples)",
    ]
    if not fixed:
        ea.append("_No corrections in the sampled Stage-1 errors._")
    for item in fixed[:8]:
        ea.append(
            f"- **true={item['true']}** before=`{item['pred']}` → after=`{item['after_pred']}`: "
            f"{item['text'][:160]}"
        )
    ea.extend(["", "### Still wrong after fine-tune (examples)"])
    if not still_wrong:
        ea.append("_None in this sample._")
    for item in still_wrong[:6]:
        ea.append(
            f"- **true={item['true']}** before=`{item['pred']}` after=`{item['after_pred']}`: "
            f"{item['text'][:160]}"
        )
    (ARTIFACTS / "error_analysis.md").write_text("\n".join(ea) + "\n")
    print(f"Wrote {ARTIFACTS / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
