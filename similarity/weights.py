"""Resolve the checkpoint for the 2nd-place RSNA-2024 Lumbar Spine Kaggle solution.

This is the model whose internal features we embed for the similarity / t-SNE work.
The checkpoint is ~0.5 GB and is **not vendored** in this repo. This module only
*locates* it; it never downloads or fabricates a source.

Resolution order:
  1. explicit `checkpoint_path` argument
  2. env var  MRI_SIM_CKPT
  3. default local path used during development (below)

The model code (the `src/` package: configs + `cnn25d_sagittal_3heads.Net`) lives in
the 2nd-place solution repository, located via `MRI_SIM_RSNA_REPO` or its default
path. If you do not have these, ask the dataset/model owner for:
  - the RSNA-2024 2nd-place solution source tree, and
  - the fold-0 stage-2 checkpoint `cfg_stage2_s2_sp1_fold0_seed813664.pt`.
"""
import os

DEFAULT_CKPT = os.path.expanduser(
    "~/Downloads/rsna-test/RSNA-2024-Competition/data/checkpoints/"
    "cfg_stage2_s2_sp1_fold0_seed813664.pt"
)
DEFAULT_RSNA_REPO = os.path.expanduser("~/Downloads/rsna-test/RSNA-2024-Competition")


def get_checkpoint_path(checkpoint_path: str | None = None) -> str:
    """Return a verified path to the RSNA-2024 2nd-place checkpoint, or raise."""
    cand = checkpoint_path or os.environ.get("MRI_SIM_CKPT") or DEFAULT_CKPT
    cand = os.path.expanduser(cand)
    if not os.path.isfile(cand):
        raise FileNotFoundError(
            "RSNA-2024 2nd-place checkpoint not found at:\n  "
            f"{cand}\n"
            "Set MRI_SIM_CKPT (or pass checkpoint_path=) to the "
            "'cfg_stage2_s2_sp1_fold0_seed813664.pt' file from the 2nd-place "
            "RSNA-2024 Lumbar Spine solution. It is not vendored here (~0.5 GB)."
        )
    return cand


def get_rsna_repo(repo_path: str | None = None) -> str:
    """Return the path to the 2nd-place solution source tree (provides `src/`)."""
    cand = repo_path or os.environ.get("MRI_SIM_RSNA_REPO") or DEFAULT_RSNA_REPO
    cand = os.path.expanduser(cand)
    if not os.path.isdir(os.path.join(cand, "src")):
        raise FileNotFoundError(
            f"RSNA-2024 solution source not found at: {cand}\n"
            "Set MRI_SIM_RSNA_REPO to the 2nd-place solution repo (must contain src/)."
        )
    return cand
