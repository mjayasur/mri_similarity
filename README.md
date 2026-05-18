# Similarity project

MRI similarity for lumbar spine: turn LumbarDISC / RSNA-2024 lumbar MRI studies into
embeddings, explore them with t-SNE, and run a Cornerstone-based **triplet** reader
study (Reference vs. A vs. B) where a clinician picks which study is more similar to
the reference.

---

## ⚠️ If you are another agent

**Please add any changes you make to the "Project structure" section of this README**
(new files, new modules, what they do). Keep it current so the next agent/person can
orient quickly.

Also: this code needs the **LumbarDISC / RSNA-2024 dataset**. Before running anything
beyond the bundled 3-study demo, ask the user for the **path to their LumbarDISC
dataset** and pass it via the `MRI_SIM_LUMBARDISC` environment variable (or the
`--lumbardisc` CLI flag where supported). Do not assume a path.

---

## Functionality so far

1. **`similarity/weights.py`** — **gets** the model + weights for whoever runs it
   (2nd-place RSNA-2024 Lumbar Spine solution, the model whose features we embed):
   - clones the model code (`github.com/brendanartley/RSNA-2024-Competition`,
     **pinned to an exact commit**) if not present;
   - downloads the ~0.5 GB checkpoint (cached) from the Kaggle dataset
     `brendanartley/rsna2024-solution-metadata` (the solution's own README hosts
     the pretrained models there). Needs free Kaggle API creds.
   - Nothing is vendored. Upstream has **no LICENSE**, so we clone (never copy)
     and pin the commit. Overrides: `MRI_SIM_CKPT`, `MRI_SIM_CKPT_URL`,
     `MRI_SIM_RSNA_GIT` (your fork), `MRI_SIM_RSNA_COMMIT`.
2. **`similarity/embeddings.py`** — loads that model and extracts per-(study, level)
   feature embeddings (spinal/foraminal/subarticular attention features concatenated),
   plus `load_precomputed()` to load embeddings we already generated for the t-SNE work.
3. **`similarity/tsne.py`** — t-SNE projection helpers and a random-triplet sampler
   (reference + near neighbor A + far point B by cosine distance).
4. **`scripts/example_tsne_triplet.py`** — **run this first.** Computes/loads
   embeddings, runs t-SNE, picks a random triplet, and saves a visualization so you
   can see the embedding space + a Reference/A/B triplet.
5. **`scripts/make_triplets.py`** — selects **40 triplets** from a LumbarDISC dataset
   (random at baseline) and writes `website/triplets.json` for the reader-study site.
5b. **`scripts/make_tsne_app.py`** — builds `website/tsne.json` for the t-SNE
   explorer: per-study t-SNE coords + severity + per-level RSNA-2024 stenosis
   findings + **exact cosine nearest neighbours** from the embeddings, and
   symlinks `website/lumbardisc` at the
   dataset image root so the DICOMs are servable.
6. **`website/`** — the start of the triplet study site (Cornerstone.js, scrollable
   DICOM stacks, slider + mouse wheel, A/B choice logged to CSV). **A collaborator
   will replace the design** — keep the data contract (`triplets.json`, `/submit`)
   stable. Ships working out of the box on the 3 bundled studies.

### Third-party persistence (important)
The model code and weights are **third-party** (Brendan Artley's 2nd-place
RSNA-2024 solution; no upstream license, so we clone + pin rather than vendor).
To survive upstream changes/deletion:
- Code: the clone is **pinned** to commit `0e795b0…`. For deletion-persistence,
  **fork** `brendanartley/RSNA-2024-Competition` to your GitHub and
  `export MRI_SIM_RSNA_GIT=git@github.com:<you>/RSNA-2024-Competition.git`.
- Weights: the checkpoint lives on a third-party Kaggle dataset. Mirror the
  `cfg_stage2_s2_sp1_fold0_seed813664.pt` somewhere you control and set
  `MRI_SIM_CKPT_URL` (or `MRI_SIM_CKPT`) to it.

### TODO (planned)
- [x] **Forked** `brendanartley/RSNA-2024-Competition` →
      `github.com/mjayasur/RSNA-2024-Competition` and set as default
      `MRI_SIM_RSNA_GIT` (deletion-persistence; not vendored — no upstream license).
- [ ] Mirror the checkpoint to owner-controlled storage; wire as default
      `MRI_SIM_CKPT_URL`.
- [ ] Add SPIDER-model embeddings to `similarity/embeddings.py` for: **Modic changes**,
      **spondylolisthesis**, **disc height**, **vertebral body height**
      (concatenate alongside the current RSNA-2024 features).
- [ ] Wire `make_triplets.py` to use embedding-based near/far sampling (not just
      random) once SPIDER embeddings land.
- [ ] Collaborator: replace `website/` front-end design (keep `triplets.json` /
      `/submit` contract).

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The model-based embedding path also needs the 2nd-place RSNA-2024 solution repo +
checkpoint. Point to it with:

```bash
export MRI_SIM_CKPT=/path/to/cfg_stage2_s2_sp1_fold0_seed813664.pt
export MRI_SIM_RSNA_REPO=/path/to/RSNA-2024-Competition      # provides src/ model code
export MRI_SIM_LUMBARDISC=/path/to/lumbardisc                 # the dataset root
```

(For the bundled demo none of these are required — precomputed embeddings / the
3 default studies are used.)

## Run the example visualization first

```bash
source .venv/bin/activate
python scripts/example_tsne_triplet.py            # uses precomputed embeddings if found
# or point at your own embeddings:
python scripts/example_tsne_triplet.py --emb /path/study_embeddings.npy --meta /path/meta.csv
```
Outputs `scripts/out/example_triplet.png` — the t-SNE with a random Reference/A/B
triplet and their MRI crops.

## Run the triplet study website

```bash
# (optional) regenerate 40 triplets from a full LumbarDISC dataset:
python scripts/make_triplets.py --lumbardisc "$MRI_SIM_LUMBARDISC" --n 40
# otherwise the bundled website/triplets.json (3 demo studies, 1 triplet) is used.

cd website && python server.py        # serves http://127.0.0.1:8077
```
### t-SNE explorer (full LumbarDISC, click a point -> similarity)

```bash
source .venv/bin/activate
python scripts/make_tsne_app.py \
    --lumbardisc /path/to/LumbarDISC/train_images   # <root>/<study>/<series>/<n>.dcm
cd website && python server.py
```
Open `http://127.0.0.1:8077/mockups/neurolens_tsne.html`. Screen 1 is the
t-SNE map (axes shown, points coloured by severity); click a study to open the
similarity screen with its exact cosine nearest neighbours.

---

Open `http://127.0.0.1:8077/`. To expose it publicly off your machine:
`cloudflared tunnel --url http://127.0.0.1:8077` (or ngrok). Responses are appended
to `website/responses.csv`.

---

## Project structure

```
mri_similarity/
├── README.md                       # this file (agents: keep this section updated)
├── requirements.txt
├── .gitignore
├── similarity/
│   ├── __init__.py
│   ├── weights.py                  # resolve 2nd-place RSNA-2024 checkpoint (path/env; not vendored)
│   ├── embeddings.py               # model load + (study,level) embeddings; load_precomputed(); TODO: SPIDER
│   └── tsne.py                     # t-SNE projection + random triplet sampler
├── scripts/
│   ├── example_tsne_triplet.py     # FIRST RUN: t-SNE + random triplet visualization
│   └── make_triplets.py            # choose 40 triplets -> website/triplets.json
├── website/                        # Cornerstone triplet reader study (collaborator will redesign)
│   ├── index.html                  # 3 scrollable DICOM stacks (slider + wheel), A/B choice
│   ├── server.py                   # static server + POST /submit -> responses.csv
│   ├── triplets.json               # triplet manifest (data contract; default = 1 demo triplet)
│   ├── tsne.json                   # generated: t-SNE coords + cosine NN + slice lists (gitignored)
│   ├── lumbardisc -> <dataset>/train_images   # generated symlink (gitignored)
│   ├── mockups/                    # design-language mockups (served at /mockups/)
│   │   ├── neurolens_similarity.html   # NeuroLens comparison layout, 3 demo studies
│   │   └── neurolens_tsne.html         # NeuroLens t-SNE map of the full LumbarDISC
│   │                                   #   set; click a point -> similarity screen
│   │                                   #   (needs tsne.json via make_tsne_app.py)
│   └── data -> ../data/default_studies   # symlink; make_triplets.py can repoint this
└── data/
    └── default_studies/            # 3 bundled LumbarDISC studies (DICOM) for the demo
        ├── 2056309275/             # normal  (Reference in demo triplet)
        ├── 7143189/                # normal  (A)
        └── 808294521/              # severe  (B)
```

_Committed with help from Claude Code._
