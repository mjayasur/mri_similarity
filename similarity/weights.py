"""Obtain the model + weights for the 2nd-place RSNA-2024 Lumbar Spine solution.

This is the model whose internal features we embed for the similarity / t-SNE
work. Provenance (verified from the solution's own README):

  * Model code  : https://github.com/brendanartley/RSNA-2024-Competition
                  (Brendan Artley's part of the 2nd-place RSNA-2024 solution)
  * Checkpoints : Kaggle dataset `brendanartley/rsna2024-solution-metadata`
                  (its README: "also contains some pretrained models ... if you
                  would just like to perform inference")

This module will, for whoever runs it:

  * `get_rsna_repo()`   — clone the (small, public, weights-free) model-code repo
                          if it is not already present.
  * `ensure_checkpoint()` — download the ~0.5 GB checkpoint from the Kaggle
                          dataset above (cached), if not already present.

The 0.5 GB checkpoint is intentionally **not vendored** in this repo. Kaggle
downloads need Kaggle API credentials (free): create a token at
https://www.kaggle.com/settings -> "Create New Token" -> save as
~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY). You must also have
accepted the RSNA-2024 competition rules on Kaggle.

Overrides (skip the download entirely):
  MRI_SIM_CKPT       absolute path to the .pt file
  MRI_SIM_CKPT_URL   direct URL to the .pt file (used if set)
  MRI_SIM_RSNA_REPO  path to an existing RSNA-2024-Competition checkout
  MRI_SIM_WEIGHTS_DIR  cache dir (default ~/.cache/mri_similarity)
"""
import os
import sys
import glob
import shutil
import subprocess
import urllib.request

CKPT_NAME = "cfg_stage2_s2_sp1_fold0_seed813664.pt"
# Upstream (Brendan Artley's 2nd-place solution). It has NO LICENSE file, so we
# do NOT vendor/copy his code; we clone it and pin to an exact commit so future
# upstream changes cannot break this pipeline. For deletion-persistence, FORK it
# to your own GitHub and set MRI_SIM_RSNA_GIT to your fork's URL.
RSNA_REPO_GIT = os.environ.get(
    "MRI_SIM_RSNA_GIT",
    "https://github.com/brendanartley/RSNA-2024-Competition.git")
RSNA_REPO_COMMIT = os.environ.get(
    "MRI_SIM_RSNA_COMMIT", "0e795b09783ec773b6996d0816b6c6ae5541c197")
KAGGLE_DATASET = "brendanartley/rsna2024-solution-metadata"

DEFAULT_CKPT = os.path.expanduser(
    f"~/Downloads/rsna-test/RSNA-2024-Competition/data/checkpoints/{CKPT_NAME}")
DEFAULT_RSNA_REPO = os.path.expanduser("~/Downloads/rsna-test/RSNA-2024-Competition")


def cache_dir() -> str:
    d = os.path.expanduser(os.environ.get("MRI_SIM_WEIGHTS_DIR",
                                          "~/.cache/mri_similarity"))
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Model code (small public repo, no weights) — clone if missing
# --------------------------------------------------------------------------- #
def get_rsna_repo(repo_path: str | None = None, auto_clone: bool = True) -> str:
    """Path to the 2nd-place solution source tree (provides `src/`).

    Clones github.com/brendanartley/RSNA-2024-Competition into the cache if it
    is not already available locally.
    """
    cand = repo_path or os.environ.get("MRI_SIM_RSNA_REPO") or DEFAULT_RSNA_REPO
    cand = os.path.expanduser(cand)
    if os.path.isdir(os.path.join(cand, "src")):
        return cand
    if not auto_clone:
        raise FileNotFoundError(
            f"RSNA-2024 solution source not found at {cand}. "
            "Set MRI_SIM_RSNA_REPO or allow auto_clone.")
    dest = os.path.join(cache_dir(), "RSNA-2024-Competition")
    if not os.path.isdir(os.path.join(dest, "src")):
        print(f"[weights] cloning model code: {RSNA_REPO_GIT} "
              f"@ {RSNA_REPO_COMMIT[:10]}", flush=True)
        subprocess.run(["git", "clone", RSNA_REPO_GIT, dest], check=True)
        try:
            subprocess.run(["git", "-C", dest, "checkout", "--quiet",
                            RSNA_REPO_COMMIT], check=True)
        except subprocess.CalledProcessError:
            print(f"[weights] WARNING: pinned commit {RSNA_REPO_COMMIT[:10]} "
                  "not found in this remote; using its default branch.",
                  flush=True)
    return dest


# --------------------------------------------------------------------------- #
# Checkpoint — resolve locally or download for the runner
# --------------------------------------------------------------------------- #
def _find_ckpt(root: str) -> str | None:
    hits = glob.glob(os.path.join(root, "**", CKPT_NAME), recursive=True)
    return hits[0] if hits else None


def _download_url(url: str, dest: str) -> str:
    print(f"[weights] downloading checkpoint from {url}", flush=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)
    return dest


def _download_kaggle(dest_dir: str) -> str:
    """Download the checkpoint via the Kaggle API dataset; return the .pt path."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "The 'kaggle' package is required to fetch the weights. "
            "`pip install kaggle`, then create an API token at "
            "https://www.kaggle.com/settings (save ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME/KAGGLE_KEY). Original error: " + repr(e))
    try:
        api = KaggleApi(); api.authenticate()
    except Exception as e:
        raise RuntimeError(
            "Kaggle authentication failed. Put kaggle.json at ~/.kaggle/ "
            "(chmod 600) or set KAGGLE_USERNAME/KAGGLE_KEY, and accept the "
            "RSNA-2024 competition rules. Error: " + repr(e))
    print(f"[weights] downloading Kaggle dataset {KAGGLE_DATASET} "
          "(~hundreds of MB, one-time)…", flush=True)
    api.dataset_download_files(KAGGLE_DATASET, path=dest_dir, unzip=True, quiet=False)
    found = _find_ckpt(dest_dir)
    if not found:
        raise FileNotFoundError(
            f"Downloaded {KAGGLE_DATASET} but '{CKPT_NAME}' was not inside it. "
            f"Inspect {dest_dir} and set MRI_SIM_CKPT to the correct .pt.")
    return found


def ensure_checkpoint(checkpoint_path: str | None = None) -> str:
    """Return a path to the checkpoint, downloading it for the runner if needed.

    Order: explicit arg -> MRI_SIM_CKPT -> local dev path -> cache ->
    MRI_SIM_CKPT_URL -> Kaggle dataset download.
    """
    for cand in (checkpoint_path, os.environ.get("MRI_SIM_CKPT"), DEFAULT_CKPT):
        if cand and os.path.isfile(os.path.expanduser(cand)):
            return os.path.expanduser(cand)

    cdir = cache_dir()
    cached = os.path.join(cdir, CKPT_NAME)
    if os.path.isfile(cached):
        return cached
    found = _find_ckpt(cdir)
    if found:
        return found

    url = os.environ.get("MRI_SIM_CKPT_URL")
    if url:
        return _download_url(url, cached)

    return _download_kaggle(cdir)


# Backwards-compatible alias
def get_checkpoint_path(checkpoint_path: str | None = None) -> str:
    return ensure_checkpoint(checkpoint_path)
