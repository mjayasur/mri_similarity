"""Lumbar MRI embeddings.

Two ways to get embeddings:

  * `load_precomputed()` — load the study-level concat embeddings we already
    generated for the t-SNE / triplet work (fast, no torch needed).
  * `load_model()` + `embed_study_levels()` — run the 2nd-place RSNA-2024 model
    and capture its spinal/foraminal/subarticular attention features.

Ported from the development script `extract_embeddings.py`.

TODO (planned): add SPIDER-model embeddings and concatenate them alongside the
RSNA-2024 features, to capture **Modic changes, spondylolisthesis, disc height,
and vertebral body height**. See README "TODO".
"""
import os
import glob
import numpy as np

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
N_FRAMES = 8
IMG_SIZE = 128
CROP_FRAC = 0.18

DEFAULT_EMB = os.path.expanduser("~/Downloads/rsna-test/data/study_embeddings_concat_heldout.npy")
DEFAULT_META = os.path.expanduser("~/Downloads/rsna-test/data/study_embeddings_concat_heldout_meta.csv")


# --------------------------------------------------------------------------- #
# Precomputed embeddings (no torch required)
# --------------------------------------------------------------------------- #
def load_precomputed(emb_path: str | None = None, meta_path: str | None = None):
    """Return (emb: np.ndarray [N, D], meta: pandas.DataFrame).

    Defaults to the study-level concat embeddings produced during development.
    """
    import pandas as pd
    emb_path = os.path.expanduser(emb_path or os.environ.get("MRI_SIM_EMB", DEFAULT_EMB))
    meta_path = os.path.expanduser(meta_path or os.environ.get("MRI_SIM_META", DEFAULT_META))
    if not os.path.isfile(emb_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Precomputed embeddings/meta not found:\n  {emb_path}\n  {meta_path}\n"
            "Pass --emb/--meta or set MRI_SIM_EMB / MRI_SIM_META, or regenerate "
            "them with the model path (see similarity.weights)."
        )
    return np.load(emb_path), pd.read_csv(meta_path)


# --------------------------------------------------------------------------- #
# Model-based embedding extraction (needs torch + the 2nd-place solution repo)
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: str | None = None, repo_path: str | None = None):
    """Load the 2nd-place RSNA-2024 model in inference mode.

    Returns (model, buf) where `buf` is a dict that the registered forward hooks
    fill with the spinal/foraminal/subarticular features on each forward pass.
    """
    import sys
    import torch
    from . import weights

    repo = weights.get_rsna_repo(repo_path)
    ckpt = weights.get_checkpoint_path(checkpoint_path)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from src.configs.cfg_stage2_s2_sp1 import cfg          # noqa: E402
    from src.models.cnn25d_sagittal_3heads import Net      # noqa: E402

    device = torch.device("cpu")
    cfg.device = device
    cfg.grad_checkpointing = False
    model = Net(cfg, pretrained=False, inference_mode=True).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    buf: dict = {}

    def _hook(name):
        def _h(_m, _i, out):
            buf[name] = out[0].detach().cpu()
        return _h

    model.attn1.register_forward_hook(_hook("spinal"))
    model.attn2.register_forward_hook(_hook("foram"))
    model.attn3.register_forward_hook(_hook("subart"))
    return model, buf


def _load_series(study_dir: str):
    import pydicom
    files = sorted(glob.glob(os.path.join(study_dir, "*.dcm")),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    out = []
    for f in files:
        ds = pydicom.dcmread(f)
        out.append((int(os.path.basename(f).split(".")[0]),
                    ds.pixel_array.astype(np.float32)))
    return out


def _crop(img, cx, cy, half, size):
    import cv2
    h, w = img.shape
    x0, y0 = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
    x1, y1 = min(w, int(round(cx + half))), min(h, int(round(cy + half)))
    p = img[y0:y1, x0:x1]
    if p.size == 0:
        return np.zeros((size, size), np.float32)
    return cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)


def embed_study_levels(model, buf, study_dir: str, level_xy: dict):
    """Concat embedding per disc level for one study.

    `level_xy` maps each of LEVELS -> (x, y) disc-center pixel coords on the
    annotated mid-sagittal instance (from train_label_coordinates.csv, condition
    'Spinal Canal Stenosis'). Returns np.ndarray of shape (5, D).
    """
    import torch
    series = _load_series(study_dir)
    if not series:
        raise RuntimeError(f"No DICOMs in {study_dir}")
    insts = [n for n, _ in series]
    # center the 8-frame window on the middle instance
    ti = len(series) // 2
    s = max(0, ti - N_FRAMES // 2)
    frames = series[s:s + N_FRAMES]
    while len(frames) < N_FRAMES:
        frames.append((frames[-1][0], np.zeros_like(frames[-1][1])))
    H, W = series[ti][1].shape
    half = CROP_FRAC * max(H, W)

    x = np.zeros((1, 5, N_FRAMES, 1, IMG_SIZE, IMG_SIZE), np.float32)
    for li, lvl in enumerate(LEVELS):
        if lvl not in level_xy:
            raise KeyError(f"missing coords for level {lvl}")
        cx, cy = level_xy[lvl]
        for fi, (_, arr) in enumerate(frames):
            patch = _crop(arr, cx, cy, half, IMG_SIZE)
            lo, hi = np.percentile(patch, [1, 99])
            x[0, li, fi, 0] = np.clip((patch - lo) / (hi - lo + 1e-6), 0, 1)

    batch = {"input": torch.from_numpy(x),
             "target": torch.zeros(1, 5, 5, 3),
             "mask": torch.ones(1, N_FRAMES, dtype=torch.bool)}
    with torch.no_grad():
        model(batch)
    es, ef, ea = buf["spinal"], buf["foram"], buf["subart"]   # each (5, 2048)
    return np.stack([np.concatenate([es[i], ef[i], ea[i]]) for i in range(5)])


def study_embedding(model, buf, study_dir: str, level_xy: dict):
    """Single study-level vector = mean of the per-level concat embeddings."""
    return embed_study_levels(model, buf, study_dir, level_xy).mean(axis=0)
