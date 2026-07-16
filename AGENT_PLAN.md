# Agent Plan — UK Headlines Classifier (Mac Studio)

**Target host:** Mac Studio, Apple M3 Ultra, 96 GB unified memory  
**Project path:** `uk_headlines/` in repo `cambridge2026`  
**Goal:** Show that a headline topic classifier scores higher on a frozen UK test set **after** UK fine-tuning than the same model trained only on general AG News headlines.

---

## Agent role

Implement this project under `uk_headlines/` on the user’s Mac Studio: environment, data (if needed), train, evaluate, write results here.

- Do **not** modify lecture notebooks (`lect2/`, `lect3/`, `lect4/`, `lect5/`) unless asked.
- Do **not** commit secrets (`api_key.txt`, `.env`).
- Prefer Hugging Face Transformers + PyTorch **MPS** (DistilBERT is small; do not use MLX-LM for this classifier).

---

## Locked stack

| Piece | Choice |
|---|---|
| Model | `distilbert-base-uncased` + 4-way classification head ([Hub](https://huggingface.co/distilbert/distilbert-base-uncased)) |
| Libraries | `torch` (MPS), `transformers`, `datasets`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `joblib` |
| General / before data | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) — title / short text prefix only |
| UK fine-tune + test | [`SetFit/bbc-news`](https://huggingface.co/datasets/SetFit/bbc-news) (BBC UK) |
| Classical baseline | Multinomial Naive Bayes + bag-of-words / unigrams |
| Metrics | **Accuracy + macro-F1 + micro-F1** on frozen UK test; per-class F1; confusion matrices |
| Fine-tuning method | **Full continued fine-tuning** (Stage 1 → Stage 2). LoRA not required for DistilBERT. |

Prepared CSVs are already in `data/` (committed). Re-run `prepare_data.py` only if regenerating.

---

## Label mapping (mandatory)

Unify both datasets to **4 labels**:

| Unified | AG News | BBC |
|---|---|---|
| `business` | Business | business |
| `sport` | Sports | sport |
| `tech` | Sci/Tech | tech |
| `politics` | World | politics |

Drop BBC `entertainment`. Document that AG `World` ↔ BBC `politics` is imperfect but intentional.

---

## Folder layout to create / complete

```text
uk_headlines/
  AGENT_PLAN.md            # this brief
  AGENT_PLAN.html          # styled copy
  README.md
  requirements.txt
  prepare_data.py          # already present
  train_stage1.py
  train_stage2.py
  train_uk_only.py
  train_nb.py
  evaluate.py
  data/                    # CSVs already present
  artifacts/
    model_ag_headlines/
    model_uk_finetuned/
    model_uk_only/
    nb_ag.joblib
    nb_uk.joblib
    metrics_before.json
    metrics_after.json
    metrics_uk_only.json
    results_table.md
    error_analysis.md
  .gitignore
```

---

## Environment setup (do first)

```bash
cd uk_headlines
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate scikit-learn pandas numpy tqdm matplotlib joblib
# Pin versions into requirements.txt after a successful install.
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

If MPS is available, use `device="mps"`. Else CPU and note it in README.

No HF API key required for these public assets. Cache under `~/.cache/huggingface/` is fine.

`.gitignore` should cover `.venv/`, `__pycache__/`, large `artifacts/model_*`, `*.pt`, etc. Prefer committing small metric JSON / results markdown, not huge weight dumps, unless asked.

---

## Data (already prepared)

| File | Role |
|---|---|
| `data/ag_train.csv` | Stage-1 train (~30k stratified AG titles) |
| `data/uk_train.csv` | UK fine-tune (70%) |
| `data/uk_val.csv` | Early stopping (15%) |
| `data/uk_test.csv` | **Frozen** test (15%) — never train on this |
| `data/split_meta.json` | Seed, sizes, notes |

Columns: `text,label` (+ `source`). Seed `42`. BBC text is a short first-sentence / token **headline proxy** when no title field exists.

To regenerate:

```bash
python prepare_data.py
```

---

## Training stages

### Stage 1 — DistilBERT on AG (“before”)

1. Init `distilbert-base-uncased`.
2. Train on `data/ag_train.csv` → 4 labels. Max length short (e.g. 64).
3. Save `artifacts/model_ag_headlines/`.
4. Eval on **`uk_test.csv` only** → `artifacts/metrics_before.json`  
   (accuracy, macro-F1, micro-F1, per-class F1).

### Stage 2 — continue fine-tune on UK (“after”)

1. Load Stage-1 checkpoint (do **not** re-init from base).
2. Train on `uk_train.csv`; early-stop on `uk_val.csv` (about 1–3 epochs).
3. Save `artifacts/model_uk_finetuned/`.
4. Eval on the **same** `uk_test.csv` → `artifacts/metrics_after.json`.

### UK only — fine-tune from base on UK (no AG)

1. Init `distilbert-base-uncased` (do **not** load Stage 1).
2. Train on `uk_train.csv`; early-stop on `uk_val.csv` (same recipe as Stage 2).
3. Save `artifacts/model_uk_only/`.
4. Eval on the **same** `uk_test.csv` → `artifacts/metrics_uk_only.json`.

This isolates whether gains come from UK data alone vs AG → UK continued training.

### Naive Bayes baseline

1. Train NB on AG titles; score UK test.
2. Retrain NB on UK train; score same UK test.
3. Save joblib models under `artifacts/`.

### Results table (`artifacts/results_table.md`)

| System | UK test accuracy | UK test macro-F1 | UK test micro-F1 |
|---|---|---|---|
| NB @ AG | … | … | … |
| DistilBERT @ AG (before) | … | … | … |
| NB @ UK | … | … | … |
| DistilBERT UK only (no AG) | … | … | … |
| DistilBERT UK fine-tuned (after) | … | … | … |

Also write `artifacts/error_analysis.md`: UK-specific terms/entities the before model misses that improve after fine-tuning.

**Note:** For single-label multi-class, micro-F1 often equals accuracy; still report both so they can be compared with macro-F1 (lect2).

---

## Suggested run order

```bash
cd uk_headlines
source .venv/bin/activate
# python prepare_data.py   # only if regenerating data
python train_nb.py
python train_stage1.py
python evaluate.py --checkpoint artifacts/model_ag_headlines --out artifacts/metrics_before.json
python train_stage2.py
python evaluate.py --checkpoint artifacts/model_uk_finetuned --out artifacts/metrics_after.json
python train_uk_only.py
python evaluate.py --checkpoint artifacts/model_uk_only --out artifacts/metrics_uk_only.json --confusion-name confusion_uk_only.png
# assemble results_table.md + error_analysis.md
```

Scripts may be one notebook if clearer; keep the same stages and artifact paths.

---

## Hard constraints

- **DO** use the same UK test set for before and after.
- **DO** report accuracy, macro-F1, and micro-F1.
- **DO** keep scripts runnable after `source .venv/bin/activate`.
- **DO NOT** train on UK test or tune on test labels.
- **DO NOT** train a large generative LLM from scratch.
- **DO NOT** commit `api_key.txt` or dump huge HF caches into git.
- **DO NOT** rewrite unrelated lecture practicals.

---

## Definition of done

- [x] Venv + `requirements.txt` work; MPS verified (or CPU noted).
- [x] UK split frozen; before/after DistilBERT metrics recorded.
- [x] UK-only DistilBERT (no AG) metrics recorded.
- [x] NB @ AG and NB @ UK metrics recorded.
- [x] `artifacts/results_table.md` shows after ≥ before on macro-F1 and micro-F1, or explains if not.
- [x] Short README / error analysis complete.

---

## Risks (acknowledge in write-up)

- `World` ↔ `politics` mapping is imperfect.
- BBC rows may be article bodies; headline proxy must stay documented.
- UK set is small — early-stop; do not over-train Stage 2.

---

## Time budget (M3 Ultra)

- Stage 1: typically under 1–2 hours  
- Stage 2: minutes  
- Naive Bayes: seconds  
Stay well under 24 hours wall clock.
