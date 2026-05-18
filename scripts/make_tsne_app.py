"""Build the data for the t-SNE -> similarity web app (website/tsne.json).

Real pipeline, no fabricated numbers:

  * embeddings  : per-study concat embeddings (RSNA-2024 2nd-place features)
  * t-SNE       : 2-D coords taken from the embedding meta (the same t-SNE used
                  in the showcase); pass --recompute to re-run t-SNE here.
  * neighbours  : exact cosine nearest neighbours from the embeddings.
  * images      : the Sagittal T2/STIR DICOM series for each study, from the
                  LumbarDISC / RSNA-2024 dataset, served via a website symlink.

Usage:

    python scripts/make_tsne_app.py                       # uses local defaults
    python scripts/make_tsne_app.py \
        --emb  /path/study_embeddings_concat_heldout.npy \
        --meta /path/study_embeddings_concat_heldout_meta.csv \
        --lumbardisc /path/to/train_images \
        --series-desc /path/train_series_descriptions.csv

`--lumbardisc` is the dataset's per-study image root (RSNA layout:
<root>/<study_id>/<series_id>/<n>.dcm). Defaults to MRI_SIM_LUMBARDISC, then
the local rsna-test copy. Do not assume a path for someone else's machine.
"""
import os
import csv
import sys
import json
import glob
import argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEF_DATA = os.path.expanduser("~/Downloads/rsna-test/data")
DEF_EMB = os.path.join(DEF_DATA, "study_embeddings_concat_heldout.npy")
DEF_META = os.path.join(DEF_DATA, "study_embeddings_concat_heldout_meta.csv")
DEF_IMGS = os.environ.get("MRI_SIM_LUMBARDISC",
                          os.path.join(DEF_DATA, "train_images"))
DEF_SDESC = os.path.join(DEF_DATA, "train_series_descriptions.csv")
LINK_NAME = "lumbardisc"  # website/<LINK_NAME> -> image root


LEVELS = [("l1_l2", "L1/L2"), ("l2_l3", "L2/L3"), ("l3_l4", "L3/L4"),
          ("l4_l5", "L4/L5"), ("l5_s1", "L5/S1")]


def _g(v):
    v = (v or "").strip()
    return v if v in ("Normal/Mild", "Moderate", "Severe") else ""


def findings(r):
    """Per-level RSNA-2024 stenosis grades, human-readable keys."""
    out = {}
    for k, lab in LEVELS:
        out[lab] = {
            "canal": _g(r.get(f"spinal_canal_stenosis_{k}")),
            "foram_l": _g(r.get(f"left_neural_foraminal_narrowing_{k}")),
            "foram_r": _g(r.get(f"right_neural_foraminal_narrowing_{k}")),
            "subart_l": _g(r.get(f"left_subarticular_stenosis_{k}")),
            "subart_r": _g(r.get(f"right_subarticular_stenosis_{k}")),
        }
    return out


def severity(sev_score, agg):
    if (agg or "").strip() == "Any Severe":
        return "severe"
    return "normal" if int(float(sev_score or 0)) == 0 else "moderate"


def sagittal_series(study_dir, want_series):
    """Pick the Sagittal T2/STIR series dir for a study (fallback: largest)."""
    if want_series:
        d = os.path.join(study_dir, str(want_series))
        if glob.glob(os.path.join(d, "*.dcm")):
            return d
    best, n = None, -1
    for sub in sorted(glob.glob(os.path.join(study_dir, "*"))):
        if os.path.isdir(sub):
            c = len(glob.glob(os.path.join(sub, "*.dcm")))
            if c > n:
                best, n = sub, c
    return best


def ordered(slice_dir):
    """RSNA instances are <int>.dcm; numeric filename sort == instance order."""
    fs = glob.glob(os.path.join(slice_dir, "*.dcm"))

    def k(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0])
        except ValueError:
            return 0
    return sorted(fs, key=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=DEF_EMB)
    ap.add_argument("--meta", default=DEF_META)
    ap.add_argument("--lumbardisc", default=DEF_IMGS,
                    help="RSNA image root: <root>/<study>/<series>/<n>.dcm")
    ap.add_argument("--series-desc", default=DEF_SDESC)
    ap.add_argument("--k", type=int, default=8, help="nearest neighbours kept")
    ap.add_argument("--recompute", action="store_true",
                    help="re-run t-SNE instead of using meta coords")
    ap.add_argument("--out", default=os.path.join(REPO, "website", "tsne.json"))
    a = ap.parse_args()

    emb = np.load(os.path.expanduser(a.emb)).astype(np.float32)
    rows = list(csv.DictReader(open(os.path.expanduser(a.meta))))
    if len(rows) != len(emb):
        sys.exit(f"meta rows ({len(rows)}) != embeddings ({len(emb)})")
    imgs = os.path.expanduser(a.lumbardisc)
    if not os.path.isdir(imgs):
        sys.exit(f"LumbarDISC image root not found: {imgs} "
                 "(set --lumbardisc or MRI_SIM_LUMBARDISC)")

    # study_id -> sagittal series id
    want = {}
    if os.path.isfile(os.path.expanduser(a.series_desc)):
        for r in csv.DictReader(open(os.path.expanduser(a.series_desc))):
            d = r.get("series_description", "")
            if "Sagittal T2" in d and r["study_id"] not in want:
                want[r["study_id"]] = r["series_id"]

    # cosine neighbours (exact): Z row-normalised -> dist = 1 - Z Zt
    from similarity.tsne import l2_normalize
    Z = l2_normalize(emb)
    D = 1.0 - (Z @ Z.T)
    np.fill_diagonal(D, np.inf)

    if a.recompute:
        from similarity.tsne import compute_tsne
        XY = compute_tsne(emb)
    else:
        XY = np.array([[float(r["x_c"]), float(r["y_c"])] for r in rows])

    points, slices, kept = [], {}, []
    for i, r in enumerate(rows):
        sid = r["study_id"]
        sdir = os.path.join(imgs, sid)
        if not os.path.isdir(sdir):
            continue
        ser = sagittal_series(sdir, want.get(sid))
        fs = ordered(ser) if ser else []
        if not fs:
            continue
        rel_ser = os.path.relpath(ser, imgs).replace(os.sep, "/")
        slices[sid] = [f"{LINK_NAME}/{rel_ser}/{os.path.basename(f)}" for f in fs]
        points.append({
            "study": sid,
            "x": round(float(XY[i, 0]), 4),
            "y": round(float(XY[i, 1]), 4),
            "sev_score": int(float(r.get("sev_score", 0) or 0)),
            "severity": severity(r.get("sev_score"), r.get("agg_severity")),
            "findings": findings(r),
        })
        kept.append(i)

    # neighbours restricted to studies we can actually display
    keep_set = set(kept)
    id_by_row = {i: rows[i]["study_id"] for i in kept}
    for p, i in zip(points, kept):
        order = [j for j in np.argsort(D[i]) if j in keep_set][:a.k]
        p["nn"] = [{"study": id_by_row[j],
                    "cos_dist": round(float(D[i, j]), 4),
                    "sim": round(float(1.0 - D[i, j]), 4)} for j in order]

    # website symlink so the server can serve the dataset DICOMs
    link = os.path.join(REPO, "website", LINK_NAME)
    if os.path.islink(link) or os.path.exists(link):
        try:
            os.remove(link)
        except IsADirectoryError:
            pass
    try:
        os.symlink(imgs, link)
    except OSError as e:
        print(f"[warn] could not create {link} -> {imgs}: {e}")

    with open(a.out, "w") as f:
        json.dump({
            "axes": {"x": "t-SNE dim 1", "y": "t-SNE dim 2"},
            "recomputed": bool(a.recompute),
            "image_root": imgs,
            "points": points,
            "slices": slices,
        }, f)
    print(f"Wrote {len(points)} studies -> {a.out}")
    print(f"website/{LINK_NAME} -> {imgs}")


if __name__ == "__main__":
    main()
