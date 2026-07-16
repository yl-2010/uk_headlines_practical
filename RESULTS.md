# UK Headlines Classifier — Results

**Visual charts:** open [`index.html`](index.html) (classification) or [`similarity.html`](similarity.html) (embeddings) in a browser.

Frozen UK BBC test set (n = 261, seed 42). All systems scored on the **same** test split.

## Main comparison

| System | Accuracy | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| Naive Bayes @ AG titles | 0.8467 | 0.8351 | 0.8467 |
| DistilBERT @ AG (before UK fine-tune) | 0.8812 | 0.8695 | 0.8812 |
| Naive Bayes @ UK train | 0.8812 | 0.8744 | 0.8812 |
| DistilBERT UK only (no AG) | 0.9464 | 0.9432 | 0.9464 |
| DistilBERT AG→UK fine-tuned (after) | **0.9579** | **0.9554** | **0.9579** |

**Deltas (AG→UK after − AG before):** macro-F1 **+0.0858**, micro-F1 **+0.0766**.

**Deltas (AG→UK after − UK only):** macro-F1 **+0.0122**, micro-F1 **+0.0115**.

UK data alone explains most of the jump over AG-only DistilBERT; continuing from AG still edges out UK-only fine-tuning on the held-out UK test set.

## Per-class F1 (DistilBERT)

| Label | AG only | UK only | AG→UK after | Δ (after − AG) |
|---|---:|---:|---:|---:|
| business | 0.8903 | 0.9427 | 0.9554 | +0.0651 |
| sport | 0.9804 | 0.9804 | 0.9804 | 0.0000 |
| tech | 0.8411 | 0.9278 | 0.9375 | +0.0964 |
| politics | 0.7664 | 0.9217 | 0.9483 | +0.1819 |

Largest gain vs AG-only: **politics** (AG `World` → BBC `politics` is a weak bridge until UK fine-tuning).

## Embedding similarity (AG vs UK)

25 headlines per label from AG train + UK (n = 200, seed 42). CLS vectors from **AG-only** and **UK-only** DistilBERT; cosine of L2-normalized category centroids. Heatmaps and chart live in [`similarity.html`](similarity.html).

**Same-label AG↔UK centroid cosine**

| Label | AG model | UK-only model |
|---|---:|---:|
| business | 0.9660 | 0.9558 |
| sport | 0.9654 | 0.8738 |
| tech | 0.9857 | 0.9644 |
| politics | 0.8928 | 0.7344 |
| Mean same-label | **0.9525** | 0.8821 |
| Mean different-label | 0.1332 | 0.2176 |

AG-only keeps same-topic AG/UK centroids closer (and different topics farther) than UK-only. Politics is the weakest bridge under both models.

## Setup notes

- Labels: `business`, `sport`, `tech`, `politics`.
- AG `World` mapped to `politics` (imperfect, intentional).
- BBC `entertainment` dropped.
- **UK only** = DistilBERT fine-tuned from `distilbert-base-uncased` on UK train (no AG Stage 1).
- **AG→UK after** = continued fine-tune from the AG Stage-1 checkpoint.
- Device: Apple MPS (Mac Studio M3 Ultra).
- Micro-F1 equals accuracy here (single-label multi-class); both reported for lect2-style comparison.

## Artifact paths

| File | Contents |
|---|---|
| [`artifacts/results_table.md`](artifacts/results_table.md) | Compact table (auto-generated) |
| [`artifacts/metrics_before.json`](artifacts/metrics_before.json) | DistilBERT @ AG metrics |
| [`artifacts/metrics_uk_only.json`](artifacts/metrics_uk_only.json) | DistilBERT UK-only metrics |
| [`artifacts/metrics_after.json`](artifacts/metrics_after.json) | DistilBERT AG→UK fine-tuned metrics |
| [`artifacts/metrics_nb.json`](artifacts/metrics_nb.json) | Naive Bayes metrics |
| [`artifacts/error_analysis.md`](artifacts/error_analysis.md) | Qualitative error examples |
| [`artifacts/confusion_before.png`](artifacts/confusion_before.png) | Confusion matrix (AG only) |
| [`artifacts/confusion_uk_only.png`](artifacts/confusion_uk_only.png) | Confusion matrix (UK only) |
| [`artifacts/confusion_after.png`](artifacts/confusion_after.png) | Confusion matrix (AG→UK) |
| [`artifacts/similarity_summary.json`](artifacts/similarity_summary.json) | Embedding similarity summary |
| [`artifacts/similarity_heatmap_ag.png`](artifacts/similarity_heatmap_ag.png) | Centroid cosine heatmap (AG model) |
| [`artifacts/similarity_heatmap_uk_only.png`](artifacts/similarity_heatmap_uk_only.png) | Centroid cosine heatmap (UK-only model) |
| [`artifacts/embed_sample.csv`](artifacts/embed_sample.csv) | Sampled headlines for embeddings |
