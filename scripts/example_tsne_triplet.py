"""FIRST-RUN example: embeddings -> t-SNE -> a random Reference/A/B triplet.

Run this to sanity-check the pipeline and see the embedding space with one
sampled triplet highlighted.

    python scripts/example_tsne_triplet.py
    python scripts/example_tsne_triplet.py --emb /path/emb.npy --meta /path/meta.csv --seed 7

Uses precomputed study-level embeddings by default (no torch needed). Writes
scripts/out/example_triplet.png.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from similarity import embeddings, tsne  # noqa: E402

HIL = {"ref": "#d7191c", "a": "#1a9641", "b": "#2c7fb8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=None, help="precomputed embeddings .npy")
    ap.add_argument("--meta", default=None, help="meta .csv (needs study_id, sev_score)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    emb, meta = embeddings.load_precomputed(a.emb, a.meta)
    print(f"Loaded {emb.shape[0]} embeddings, dim {emb.shape[1]}")

    xy = tsne.compute_tsne(emb, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    t = tsne.random_triplet(emb, rng=rng, near_far=True)

    def sid(i):
        return int(meta.iloc[i]["study_id"]) if "study_id" in meta.columns else i
    sev = (meta["sev_score"].values if "sev_score" in meta.columns
           else np.zeros(len(meta)))
    print(f"Reference: study {sid(t['ref'])}")
    print(f"A (near):  study {sid(t['a'])}  cosine d = {t['d_a']:.3f}")
    print(f"B (far):   study {sid(t['b'])}  cosine d = {t['d_b']:.3f}")

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=160)
    rank = (np.argsort(np.argsort(sev)).astype(float) / max(1, len(sev) - 1))
    ax.scatter(xy[:, 0], xy[:, 1], c=rank, cmap="viridis", s=22,
               alpha=0.7, linewidth=0, rasterized=True)
    for key in ("ref", "a", "b"):
        i = t[key]
        ax.scatter(xy[i, 0], xy[i, 1], s=360, facecolor="none",
                   edgecolor=HIL[key], linewidth=3, zorder=6)
        ax.annotate(key.upper(), (xy[i, 0], xy[i, 1]),
                    xytext=(xy[i, 0], xy[i, 1] + 2.4), ha="center",
                    fontsize=11, fontweight="bold", color=HIL[key])
    ax.text(0.02, 0.98,
            f"REF → A (near): cosine d = {t['d_a']:.2f}\n"
            f"REF → B (far):  cosine d = {t['d_b']:.2f}",
            transform=ax.transAxes, va="top", family="monospace",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.4"))
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("Embedding space + a random Reference / A / B triplet",
                 fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "out", "example_triplet.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
