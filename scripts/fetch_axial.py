"""Backfill Axial T2 DICOMs for the triplet studies from the RSNA-2024 Kaggle
competition, then regenerate triplets.json and resync to Tower.

The local LumbarDISC copy is a Sagittal-T2-only subset, so the per-panel
plane toggle has nothing to switch to until axial is present. This script
pulls the Axial T2 series (instance by instance) for every study used in
website/triplets.json, drops them into the existing train_images tree, then
re-runs make_triplets.py (which now emits per-plane `views`) and rsyncs the
new DICOMs + triplets.json to the Tower deployment.

    source ~/.venv/bin/activate
    python scripts/fetch_axial.py            # long-running; safe to background

Idempotent: files already on disk are skipped.
"""
import os
import csv
import json
import time
import glob
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser("~/Downloads/rsna-test/data")
IMGS = os.path.join(DATA, "train_images")
SDESC = os.path.join(DATA, "train_series_descriptions.csv")
TRIP = os.path.join(REPO, "website", "triplets.json")
LOG = os.path.join(DATA, "axial_fetch.log")
COMP = "rsna-2024-lumbar-spine-degenerative-classification"
MISS_TOL = 3      # consecutive missing instance numbers => end of series
MAX_INST = 400    # hard safety cap per series


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    studies = set()
    for t in json.load(open(TRIP))["triplets"]:
        for k in ("ref", "a", "b"):
            studies.add(t[k]["study"])

    axial = {}
    for r in csv.DictReader(open(SDESC)):
        if r["study_id"] in studies and \
           (r.get("series_description") or "").strip() == "Axial T2":
            axial.setdefault(r["study_id"], []).append(r["series_id"])

    log(f"start: {len(studies)} studies, "
        f"{sum(len(v) for v in axial.values())} axial series")
    got = skip = 0
    for si, (study, sers) in enumerate(sorted(axial.items()), 1):
        for ser in sers:
            dest = os.path.join(IMGS, study, str(ser))
            os.makedirs(dest, exist_ok=True)
            miss = 0
            for n in range(1, MAX_INST + 1):
                fp = os.path.join(dest, f"{n}.dcm")
                if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                    got += 1
                    miss = 0
                    continue
                fn = f"train_images/{study}/{ser}/{n}.dcm"
                try:
                    api.competition_download_file(COMP, fn, path=dest,
                                                  force=True, quiet=True)
                    if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                        got += 1
                        miss = 0
                    else:
                        miss += 1
                except Exception:
                    miss += 1
                if miss >= MISS_TOL:
                    break
                time.sleep(0.05)
        if si % 10 == 0 or si == len(axial):
            log(f"{si}/{len(axial)} studies · {got} dcm fetched")
    log(f"download done: {got} fetched, {skip} pre-existing")

    # regenerate triplets.json (now with Axial T2 views) and resync to Tower
    subprocess.run([
        "python", os.path.join(REPO, "scripts", "make_triplets.py"),
        "--lumbardisc", IMGS, "--n", "40"], check=True, cwd=REPO)
    log("regenerated triplets.json")

    with open("/tmp/triplet_studies.txt", "w") as f:
        f.write("\n".join(sorted(studies)) + "\n")
    subprocess.run(
        f"cd {IMGS} && tar -cf - -T /tmp/triplet_studies.txt | "
        "ssh -o BatchMode=yes tower "
        "'tar -xf - -C ~/mri_similarity/site_data/train_images "
        "&& find ~/mri_similarity/site_data/train_images -name \"._*\" -delete'",
        shell=True, check=False)
    subprocess.run(
        ["rsync", "-az", "-e", "ssh -o BatchMode=yes", TRIP,
         os.path.join(REPO, "website", "index.html"),
         "tower:~/mri_similarity/website/"], check=False)
    log("resynced DICOMs + triplets.json + index.html to Tower — DONE")


if __name__ == "__main__":
    main()
