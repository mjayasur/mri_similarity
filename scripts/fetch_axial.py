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
# All paths are env-overridable so this runs either on the Mac (default,
# then rsyncs to Tower) or natively ON Tower (set AXIAL_* + AXIAL_NO_SYNC=1
# so the daily cron needs no SSH back out).
IMGS = os.path.expanduser(os.environ.get("AXIAL_IMGS",
                                         os.path.join(DATA, "train_images")))
SDESC = os.path.expanduser(os.environ.get(
    "AXIAL_SDESC", os.path.join(DATA, "train_series_descriptions.csv")))
TRIP = os.path.expanduser(os.environ.get(
    "AXIAL_TRIP", os.path.join(REPO, "website", "triplets.json")))
LOG = os.path.expanduser(os.environ.get(
    "AXIAL_LOG", os.path.join(DATA, "axial_fetch.log")))
MKTRIP = os.path.expanduser(os.environ.get(
    "AXIAL_MKTRIP", os.path.join(REPO, "scripts", "make_triplets.py")))
PYBIN = os.environ.get("AXIAL_PY", "python")
NO_SYNC = os.environ.get("AXIAL_NO_SYNC") == "1"
COMP = "rsna-2024-lumbar-spine-degenerative-classification"
MISS_TOL = 3      # consecutive genuine 404s => end of series
MAX_INST = 400    # hard safety cap per series
# Kaggle throttles competition single-file downloads per ACCOUNT (a daily-ish
# quota -- confirmed identical 429 from two different IPs / clients). So this
# is a resumable DAILY job: fetch a bounded number of files per run, paced
# politely, and bail fast when the quota is clearly closed.
MAX_FILES = int(os.environ.get("AXIAL_MAX_FILES", "350"))   # new files per run
BASE_SLEEP = float(os.environ.get("AXIAL_BASE_SLEEP", "4"))  # sec between files
MAX_RETRY = 3     # 429 backoff attempts per instance before declaring "wall"
WALL_TOL = 3      # instances that hit the wall in a row => quota closed, stop


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
    have = 0          # files already on disk (resumed)
    new = 0           # files fetched this run
    wall_streak = 0   # consecutive instances that exhausted retries on 429
    stop = None       # 'cap' | 'wall' | None(=finished sweep)
    for si, (study, sers) in enumerate(sorted(axial.items()), 1):
        if stop:
            break
        for ser in sers:
            dest = os.path.join(IMGS, study, str(ser))
            os.makedirs(dest, exist_ok=True)
            miss = 0
            for n in range(1, MAX_INST + 1):
                fp = os.path.join(dest, f"{n}.dcm")
                if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                    have += 1
                    miss = 0
                    continue
                fn = f"train_images/{study}/{ser}/{n}.dcm"
                back, tries, r = 10, 0, "retry"
                while r == "retry" and tries < MAX_RETRY:
                    r = fetch_one(api, fn, dest)
                    if r == "retry":
                        tries += 1
                        time.sleep(back)
                        back = min(int(back * 2), 60)
                if r is True:
                    new += 1
                    miss = 0
                    wall_streak = 0
                elif r is False:                  # genuine 404 -> series end
                    miss += 1
                    wall_streak = 0
                else:                             # still 429 after MAX_RETRY
                    wall_streak += 1
                    log(f"throttle wall at {study}/{ser}/{n} "
                        f"(streak {wall_streak}/{WALL_TOL})")
                if wall_streak >= WALL_TOL:
                    stop = "wall"
                    break
                if new >= MAX_FILES:
                    stop = "cap"
                    break
                if miss >= MISS_TOL:
                    break
                time.sleep(BASE_SLEEP)
            if stop:
                break
        if si % 5 == 0 or si == len(axial) or stop:
            log(f"{si}/{len(axial)} studies · +{new} new, {have} present")
    log(f"run end ({stop or 'swept all'}): +{new} new this run")

    # regenerate triplets.json (now with Axial T2 views)
    subprocess.run([PYBIN, MKTRIP, "--lumbardisc", IMGS,
                    "--n", "40", "--out", TRIP], check=True)
    log(f"regenerated {TRIP}")

    if NO_SYNC:
        log("running on the serving host — no resync needed; DONE")
        return

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
