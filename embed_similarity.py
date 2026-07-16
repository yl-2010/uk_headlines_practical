"""Sample AG/UK headlines, extract DistilBERT CLS embeddings, save similarity viz."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["business", "sport", "tech", "politics"]
GROUP_ORDER = [f"AG-{lab}" for lab in LABELS] + [f"UK-{lab}" for lab in LABELS]
N_PER_LABEL = 25
SEED = 42
MAX_LEN = 64
BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

CHECKPOINTS = {
    "ag": ARTIFACTS / "model_ag_headlines",
    "uk_only": ARTIFACTS / "model_uk_only",
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_headlines() -> pd.DataFrame:
    ag = pd.read_csv(DATA_DIR / "ag_train.csv")
    uk = pd.read_csv(DATA_DIR / "uk_all.csv")

    def take(df: pd.DataFrame, source: str) -> pd.DataFrame:
        parts = [
            g.sample(n=N_PER_LABEL, random_state=SEED)
            for _, g in df.groupby("label", sort=False)
        ]
        out = pd.concat(parts, ignore_index=True)[["text", "label"]].copy()
        out["source"] = source
        return out

    sample = pd.concat([take(ag, "AG"), take(uk, "UK")], ignore_index=True)
    sample["group"] = sample["source"] + "-" + sample["label"]
    return sample


@torch.no_grad()
def embed_cls(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model.distilbert(**enc).last_hidden_state
        cls = hidden[:, 0]
        cls = F.normalize(cls, p=2, dim=1)
        chunks.append(cls.cpu().numpy())
    return np.vstack(chunks)


def group_centroids(embeddings: np.ndarray, groups: list[str]) -> np.ndarray:
    cents = []
    for name in GROUP_ORDER:
        idx = [i for i, g in enumerate(groups) if g == name]
        mean = embeddings[idx].mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        cents.append(mean)
    return np.vstack(cents)


def cosine_matrix(centroids: np.ndarray) -> np.ndarray:
    return centroids @ centroids.T


def same_label_cross_domain(matrix: np.ndarray) -> dict[str, float]:
    """Cosine between AG-{label} and UK-{label} for each label."""
    out = {}
    for i, lab in enumerate(LABELS):
        ag_i = i
        uk_i = i + len(LABELS)
        out[lab] = float(matrix[ag_i, uk_i])
    return out


def cross_domain_means(matrix: np.ndarray) -> dict[str, float]:
    """Mean same-label vs different-label AG↔UK block cosines."""
    n = len(LABELS)
    block = matrix[:n, n:]
    same = float(np.mean(np.diag(block)))
    mask = ~np.eye(n, dtype=bool)
    different = float(np.mean(block[mask]))
    return {"mean_same_label": same, "mean_different_label": different}


def save_heatmap(matrix: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(matrix, cmap="Blues", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(GROUP_ORDER, fontsize=8)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def analyze_checkpoint(
    name: str,
    checkpoint: Path,
    sample: pd.DataFrame,
    device: torch.device,
) -> dict:
    print(f"embedding with {checkpoint.name} …")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)

    texts = sample["text"].astype(str).tolist()
    groups = sample["group"].tolist()
    emb = embed_cls(model, tok, texts, device)
    cents = group_centroids(emb, groups)
    matrix = cosine_matrix(cents)
    same = same_label_cross_domain(matrix)
    means = cross_domain_means(matrix)

    heatmap_name = f"similarity_heatmap_{name}.png"
    save_heatmap(
        matrix,
        ARTIFACTS / heatmap_name,
        title=f"Category centroid cosine — {checkpoint.name}",
    )

    return {
        "checkpoint": str(checkpoint),
        "heatmap": heatmap_name,
        "group_order": GROUP_ORDER,
        "matrix": [[round(float(x), 4) for x in row] for row in matrix],
        "same_label_ag_uk": {k: round(v, 4) for k, v in same.items()},
        "cross_domain": {k: round(v, 4) for k, v in means.items()},
    }


def main() -> None:
    device = get_device()
    print(f"device={device}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    sample = sample_headlines()
    sample_path = ARTIFACTS / "embed_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"wrote {sample_path} (n={len(sample)})")

    models = {}
    for key, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        models[key] = analyze_checkpoint(key, path, sample, device)

    summary = {
        "n_per_label": N_PER_LABEL,
        "seed": SEED,
        "n_headlines": int(len(sample)),
        "labels": LABELS,
        "group_order": GROUP_ORDER,
        "sample_csv": "embed_sample.csv",
        "models": models,
    }
    out = ARTIFACTS / "similarity_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")
    for key, payload in models.items():
        print(key, payload["same_label_ag_uk"], payload["cross_domain"])


if __name__ == "__main__":
    main()
