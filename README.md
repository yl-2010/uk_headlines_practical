# UK Headlines Practical

Domain adaptation for topic classification: DistilBERT trained on AG News titles, evaluated on a frozen UK BBC test set, then fine-tuned on UK data and re-evaluated (accuracy, macro-F1, micro-F1). Also compares a UK-only DistilBERT (fine-tuned from base on UK, no AG) and Multinomial Naive Bayes baselines.

## Live results (GitHub Pages)

**Site:** [https://yl-2010.github.io/uk_headlines_practical/](https://yl-2010.github.io/uk_headlines_practical/)

| Page | File |
|------|------|
| Classification (home) | [`index.html`](index.html) |
| Embedding similarity | [`similarity.html`](similarity.html) |
| Headline lengths | [`lengths.html`](lengths.html) |
| Novel words | [`novel-words.html`](novel-words.html) |
| About (hardware, model, fine-tuning) | [`about.html`](about.html) |

Local preview: open `index.html` in a browser (relative links to `artifacts/` images must stay next to the HTML files).

## Setup

```bash
cd uk_headlines_practical
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

## Run order

```bash
source .venv/bin/activate
# Data CSVs are already in data/; only regenerate if needed:
# python prepare_data.py
python train_nb.py
python train_stage1.py
python evaluate.py --checkpoint artifacts/model_ag_headlines --out artifacts/metrics_before.json
python train_stage2.py
python evaluate.py --checkpoint artifacts/model_uk_finetuned --out artifacts/metrics_after.json
python train_uk_only.py
python evaluate.py --checkpoint artifacts/model_uk_only --out artifacts/metrics_uk_only.json --confusion-name confusion_uk_only.png
python assemble_results.py
python embed_similarity.py
python novel_words.py
```

## Data notes

- **AG News** (`fancyzhx/ag_news`): title / short prefix; stratified ~30k subset for Stage 1.
- **BBC UK** (`SetFit/bbc-news`): drop `entertainment`; stratified 70/15/15 train/val/test, seed 42. Test is frozen.
- **Labels:** `business`, `sport`, `tech`, `politics`. AG `World` → `politics` (imperfect, intentional).
- **BBC headline proxy:** use `title`/`headline` if present; else first sentence or first ~16 tokens of the article body.

## Artifacts

| Path | Contents |
|---|---|
| `data/*.csv` | Prepared `text,label` splits |
| `artifacts/model_ag_headlines/` | Stage-1 DistilBERT |
| `artifacts/model_uk_finetuned/` | Stage-2 UK fine-tune (from AG) |
| `artifacts/model_uk_only/` | DistilBERT fine-tuned on UK only (no AG) |
| `artifacts/metrics_*.json` | Accuracy + macro-F1 + micro-F1 on UK test |
| `artifacts/results_table.md` | Comparison table |
| `artifacts/error_analysis.md` | Short qualitative notes |
| `artifacts/similarity_summary.json` | AG/UK category-centroid cosines |
| `artifacts/similarity_heatmap_*.png` | Centroid cosine heatmaps |
| `artifacts/novel_words_summary.json` | UK-test words missing from UK / AG train |

Model weights are gitignored; metrics, charts, and HTML results are committed for the Pages site.

See also [`AGENT_PLAN.md`](AGENT_PLAN.md) / [`AGENT_PLAN.html`](AGENT_PLAN.html) and [`RESULTS.md`](RESULTS.md).
