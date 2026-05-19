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
import sys
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
MISS_TOL = 3      # consecutive genuine 404s => end of series
MAX_INST = 400    # hard safety cap per series
MAX_HOURS = float(os.environ.get("AXIAL_MAX_HOURS", "5"))  # wall-clock guard
BASE_SLEEP = 0.7  # polite pace between files (avoid re-tripping the limit)


def _status(e):
    return (getattr(e, "status", None)
            or getattr(getattr(e, "response", None), "status_code", None))


def fetch_one(api, fn, dest):
    """Return True (saved), False (genuine 404 -> series end), or 'retry'.

    429 / 403 / 5xx are throttling/transient: caller backs off and retries the
    SAME instance (never counted as a miss), so a rate limit can't truncate a
    series the way the first run did.
    """
    b = os.path.basename(fn)
    fp = os.path.join(dest, b)
    try:
        api.competition_download_file(COMP, fn, path=dest,
                                      force=True, quiet=True)
        return os.path.isfile(fp) and os.path.getsize(fp) > 0
    except Exception as e:
        st = _status(e)
        if st == 404:
            return False
        return "retry"   # 429/403/5xx/other -> transient


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
    got = 0
    deadline = time.time() + MAX_HOURS * 3600
    out_of_time = False
    for si, (study, sers) in enumerate(sorted(axial.items()), 1):
        if out_of_time:
            break
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
                back = 30
                while True:                       # retry SAME instance on 429
                    if time.time() > deadline:
                        out_of_time = True
                        break
                    r = fetch_one(api, fn, dest)
                    if r is True:
                        got += 1
                        miss = 0
                        break
                    if r is False:                # genuine 404 -> series end
                        miss += 1
                        break
                    log(f"throttled (429) at {study}/{ser}/{n}; "
                        f"sleeping {back}s")
                    time.sleep(back)
                    back = min(int(back * 2), 600)
                if out_of_time or miss >= MISS_TOL:
                    break
                time.sleep(BASE_SLEEP)
            if out_of_time:
                break
        if si % 5 == 0 or si == len(axial) or out_of_time:
            log(f"{si}/{len(axial)} studies · {got} dcm present")
    log(f"download {'PAUSED (time cap, resumable)' if out_of_time else 'done'}"
        f": {got} dcm present")

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
