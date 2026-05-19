"""Select triplets for the reader-study website and write website/triplets.json.

Baseline = random triplets (Reference, A, B). Each study contributes its DICOM
slices in instance order so Cornerstone can scroll the stack. Per study we emit
**every imaging plane present** (LumbarDISC has Sagittal T2/STIR, Sagittal T1,
Axial T2 -- there is no coronal series in this dataset), so the website can let
the reader toggle the plane per panel.

    # default: 1 triplet from the 3 bundled studies
    python scripts/make_triplets.py

    # 40 random triplets from a full LumbarDISC dataset
    python scripts/make_triplets.py --lumbardisc "$MRI_SIM_LUMBARDISC" --n 40

triplets.json schema (additive -- `slices` kept for back-compat):
    { triplets: [ { id, ref:{study, slices:[...], default_plane,
                     views:{ "Sagittal T2":[...], "Axial T2":[...] }},
                     a:{...}, b:{...} } ] }
`slices` == the default plane (Sagittal T2/STIR if present, else the largest).

A study folder may directly contain *.dcm, or contain series subfolders. If a
train_series_descriptions.csv is found (next to the dataset root) it is used to
label series by plane; otherwise the largest *.dcm series is the only plane.
"""
import os
import sys
import csv
import glob
import json
import argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(REPO, "data", "default_studies")

# series_description -> short plane label shown in the UI
PLANE = {
    "Sagittal T2/STIR": "Sagittal T2",
    "Sagittal T1": "Sagittal T1",
    "Axial T2": "Axial T2",
}
DEFAULT_PLANE_ORDER = ["Sagittal T2", "Axial T2", "Sagittal T1"]


def _series_desc_csv(root):
    for c in (os.path.join(root, "train_series_descriptions.csv"),
              os.path.join(os.path.dirname(root.rstrip("/")),
                           "train_series_descriptions.csv")):
        if os.path.isfile(c):
            return c
    return None


def _ordered(slice_dir):
    """DICOM paths in instance order (RSNA instances are <int>.dcm)."""
    fs = glob.glob(os.path.join(slice_dir, "*.dcm"))

    def k(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0])
        except ValueError:
            return 0
    return sorted(fs, key=k)


def _largest_series(study_dir):
    if glob.glob(os.path.join(study_dir, "*.dcm")):
        return study_dir
    best, n = None, -1
    for sub in sorted(glob.glob(os.path.join(study_dir, "*"))):
        if os.path.isdir(sub):
            c = len(glob.glob(os.path.join(sub, "*.dcm")))
            if c > n:
                best, n = sub, c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lumbardisc", default=os.environ.get("MRI_SIM_LUMBARDISC"),
                    help="dataset root with per-study folders (default: bundled 3)")
    ap.add_argument("--data", default=None, help="explicit study-folders root")
    ap.add_argument("--series-desc", default=None,
                    help="train_series_descriptions.csv (auto-detected if omitted)")
    ap.add_argument("--n", type=int, default=40, help="number of triplets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "website", "triplets.json"))
    a = ap.parse_args()

    root = os.path.expanduser(a.data or a.lumbardisc or DEFAULT_DATA)
    sdesc = a.series_desc or _series_desc_csv(root)

    # study_id -> { plane_label -> series_id }
    plane_series = {}
    if sdesc and os.path.isfile(sdesc):
        for r in csv.DictReader(open(sdesc)):
            lab = PLANE.get((r.get("series_description") or "").strip())
            if lab:
                plane_series.setdefault(r["study_id"], {}).setdefault(
                    lab, r["series_id"])

    studies = [d for d in sorted(glob.glob(os.path.join(root, "*")))
               if os.path.isdir(d) and _largest_series(d)]
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
        pass

    def views_for(study_path):
        sid = os.path.basename(study_path)
        out = {}
        for lab, ser in plane_series.get(sid, {}).items():
            sd = os.path.join(study_path, str(ser))
            files = _ordered(sd)
            if files:
                rel = os.path.relpath(sd, root)
                out[lab] = ["data/" + os.path.join(
                    rel, os.path.basename(f)).replace(os.sep, "/")
                    for f in files]
        if not out:  # no series csv / unlabeled -> single plane = largest series
            sd = _largest_series(study_path)
            rel = os.path.relpath(sd, root)
            out["Sagittal T2"] = ["data/" + os.path.join(
                rel, os.path.basename(f)).replace(os.sep, "/")
                for f in _ordered(sd)]
        return out

    def entry(study_path):
        sid = os.path.basename(study_path)
        views = views_for(study_path)
        default = next((p for p in DEFAULT_PLANE_ORDER if p in views),
                       next(iter(views)))
        return {"study": sid, "slices": views[default],
                "default_plane": default, "views": views}

    rng = np.random.default_rng(a.seed)
    idx = list(range(len(studies)))
    triplets = []
    if len(idx) == 3:
        r, x, y = idx
        triplets.append({"id": 1, "ref": entry(studies[r]),
                         "a": entry(studies[x]), "b": entry(studies[y])})
    else:
        rng.shuffle(idx)
        n_trip = min(a.n, len(idx) // 3)
        for k in range(n_trip):
            r, x, y = idx[3 * k: 3 * k + 3]
            triplets.append({"id": k + 1, "ref": entry(studies[r]),
                             "a": entry(studies[x]), "b": entry(studies[y])})

    with open(a.out, "w") as f:
        json.dump({"data_root": root, "triplets": triplets}, f, indent=1)
    planes = sorted({p for t in triplets for s in ("ref", "a", "b")
                     for p in t[s]["views"]})
    print(f"Wrote {len(triplets)} triplet(s) -> {a.out}")
    print(f"planes present: {planes}  (data root: {root})")


if __name__ == "__main__":
    main()
