"""t-SNE projection and triplet sampling over study-level embeddings."""
import numpy as np


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def cosine_dist(emb_n: np.ndarray, i: int, j: int) -> float:
    return 1.0 - float(emb_n[i] @ emb_n[j])


def compute_tsne(emb: np.ndarray, perplexity: float = 30.0,
                  seed: int = 0) -> np.ndarray:
    """Return (N, 2) t-SNE coordinates for the embedding matrix."""
    from sklearn.manifold import TSNE
    n = len(emb)
    perp = float(min(perplexity, max(5, (n - 1) // 3)))
    ts = TSNE(n_components=2, perplexity=perp, init="pca",
              random_state=seed, metric="cosine")
    return ts.fit_transform(emb)


def random_triplet(emb: np.ndarray, rng: np.random.Generator | None = None,
                    near_far: bool = True):
    """Pick a (reference, A, B) triplet of row indices.

    If `near_far`, A is a genuine near neighbor (small cosine distance) and B is
    a far point, so the triplet is interpretable; otherwise all three are random.
    Returns dict(ref=int, a=int, b=int, d_a=float, d_b=float).
    """
    rng = rng or np.random.default_rng()
    n = len(emb)
    emb_n = l2_normalize(emb)
    ref = int(rng.integers(n))
    if not near_far:
        a, b = (int(x) for x in rng.choice([k for k in range(n) if k != ref],
                                           size=2, replace=False))
    else:
        d = np.array([cosine_dist(emb_n, ref, j) if j != ref else np.inf
                      for j in range(n)])
        order = np.argsort(d)
        a = int(order[0])                       # nearest neighbor
        b = int(np.argsort(-d)[0])              # farthest point
    return {"ref": ref, "a": a, "b": b,
            "d_a": cosine_dist(emb_n, ref, a),
            "d_b": cosine_dist(emb_n, ref, b)}
