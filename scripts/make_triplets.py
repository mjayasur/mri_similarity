"""Select triplets for the reader-study website and write website/triplets.json.

Baseline = random triplets (Reference, A, B). Each study contributes its
mid-sagittal DICOM slices in instance order so Cornerstone can scroll the stack.

    # default: 1 triplet from the 3 bundled studies
    python scripts/make_triplets.py

    # 40 random triplets from a full LumbarDISC dataset
    python scripts/make_triplets.py --lumbardisc "$MRI_SIM_LUMBARDISC" --n 40

A study folder may either directly contain *.dcm, or contain series subfolders
(the largest *.dcm series is used).

TODO: replace random selection with embedding near/far sampling once SPIDER
embeddings are available (see README).
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(REPO, "data", "default_studies")


def _slice_dir(study_dir):
    """Return the dir holding the .dcm slices for a study (handles series subdirs)."""
    if glob.glob(os.path.join(study_dir, "*.dcm")):
        return study_dir
    best, n = None, -1
    for sub in sorted(glob.glob(os.path.join(study_dir, "*"))):
        if os.path.isdir(sub):
            c = len(glob.glob(os.path.join(sub, "*.dcm")))
            if c > n:
                best, n = sub, c
    return best


def _ordered_slices(slice_dir):
    """DICOM file paths sorted by InstanceNumber (fallback: filename number)."""
    import pydicom
    files = glob.glob(os.path.join(slice_dir, "*.dcm"))

    def key(p):
        try:
            return int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber)
        except Exception:
            try:
                return int(os.path.basename(p).split(".")[0])
            except Exception:
                return 0
    return sorted(files, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lumbardisc", default=os.environ.get("MRI_SIM_LUMBARDISC"),
                    help="dataset root with per-study folders (defaults to bundled 3 studies)")
    ap.add_argument("--data", default=None, help="explicit study-folders root")
    ap.add_argument("--n", type=int, default=40, help="number of triplets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "website", "triplets.json"))
    a = ap.parse_args()

    root = a.data or a.lumbardisc or DEFAULT_DATA
    root = os.path.expanduser(root)
    studies = [d for d in sorted(glob.glob(os.path.join(root, "*")))
               if os.path.isdir(d) and _slice_dir(d)]
    if len(studies) < 3:
        sys.exit(f"Need >=3 studies with DICOMs under {root}; found {len(studies)}.")

    # point website/data at the chosen dataset root so the server can serve it
    link = os.path.join(REPO, "website", "data")
    if os.path.islink(link) or os.path.exists(link):
        try:
            os.remove(link)
        except IsADirectoryError:
            pass
    try:
        os.symlink(os.path.relpath(root, os.path.join(REPO, "website")), link)
    except OSError:
        pass  # leave existing default symlink if we cannot relink

    rng = np.random.default_rng(a.seed)
    idx = list(range(len(studies)))
    n_trip = min(a.n, len(idx) // 3) if len(idx) > 3 else 1

    def entry(study_path):
        sid = os.path.basename(study_path)
        sd = _slice_dir(study_path)
        rel = os.path.relpath(sd, root)
        slices = ["data/" + os.path.join(rel, os.path.basename(f)).replace(os.sep, "/")
                  for f in _ordered_slices(sd)]
        return {"study": sid, "slices": slices}

    triplets = []
    if len(idx) == 3:
        r, x, y = idx
        triplets.append({"id": 1, "ref": entry(studies[r]),
                         "a": entry(studies[x]), "b": entry(studies[y])})
    else:
        rng.shuffle(idx)
        for k in range(n_trip):
            r, x, y = idx[3 * k: 3 * k + 3]
            triplets.append({"id": k + 1, "ref": entry(studies[r]),
                             "a": entry(studies[x]), "b": entry(studies[y])})

    with open(a.out, "w") as f:
        json.dump({"data_root": root, "triplets": triplets}, f, indent=1)
    print(f"Wrote {len(triplets)} triplet(s) -> {a.out}  (data root: {root})")


if __name__ == "__main__":
    main()
