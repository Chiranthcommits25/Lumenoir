# Datasets

Status checked on 6 November 2026 with `python -m training.data_pipeline check`:

- Local data is present under `datasets/raw`: LOL-v1, LOL-v2 Real/Synthetic,
  `SICE_Paired`, and the extracted additional NPE/LIME/MEF/VV folders.
- ExDark and DICM are not present locally.
- Nothing is downloaded automatically by this repository.

The dataset strategy, counts and seeds come from `final_decision.md` §1. This
document covers the mechanics; the decision document remains authoritative.

## Expected layout

Place each dataset under `datasets/raw/<dataset>/` (git-ignored). The pipeline
never modifies or deletes these files.

```
datasets/raw/
├── lol_v1/
│   ├── our485/{low,high}/             # 485 pairs
│   └── eval15/{low,high}/             # 15 pairs
├── lol_v2_real/
│   └── Real_captured/{Train,Test}/{low,high}/   # 689 train / 100 test
├── lol_v2_syn/
│   └── Synthetic/{Train,Test}/{low,high}/       # 900 train / 100 test
├── sice/
│   ├── part1/                         # 3022 SICE Part 1 images (flat)
│   ├── part2_lowlight/                # curated 767 low-light Part 2 images
│   └── do_no_harm/                    # 100 frames at 0EV and +1EV
├── dicm/                              # 64 images
├── lime/                              # 10 images
├── mef/                               # 17 images
├── npe/                               # 85 images (NPE + NPE-ex1/ex2/ex3)
├── vv/                                # 24 images
├── exdark/<lighting-condition>/       # 7363 images across 10 conditions
└── own/
    ├── extremely_dark/                # 20 captures
    ├── non_uniform/                   # 20 captures
    ├── normal/                        # 20 captures
    └── paired/{low,high}/             # about 20 pseudo-reference scenes
```

Low/reference folders are matched case-insensitively (`low/Low/input`,
`high/High/target/gt/normal`), and pairs are matched by filename stem. Anything
that does not pair is reported, not silently dropped.

SICE needs curated folders because the release does not label which exposure is
"low-light". Put the 767 Image-from-Part-2 low-light images in
`part2_lowlight/` and the 100 well-lit frames in `do_no_harm/`. The pipeline
checks the counts and refuses to guess.

## Split protocol

| Pool | Composition | Count |
| --- | --- | ---: |
| Fine-tune | SICE P1 train 2422 + LOL-v1 train 485 + LOL-v2-Real train minus 100 val | 3496 |
| Validation | SICE P1 val 600 + 100 LOL-v2-Real train + 100 LOL-v2-Syn train | 800 |
| Test (with GT) | LOL-v1 15 + LOL-v2-Real 100 + LOL-v2-Syn 100 + SICE P2 200 of 767 | 415 |
| Unseen (no GT) | DICM 64, LIME 10, MEF 17, NPE 85, VV 24, ExDark 20×10, own 60 | 460 |
| Do-no-harm | SICE 0EV/+1EV frames | 100 |

- Seed 42 for every sampled subset, matching `final_decision.md`.
- Zero-DCE's Part 1 split (3022 → 2422/600) has no published seed; seed 42 is
  used and recorded in `training/splits/split_report.json`.
- The extreme-dark training set is generated on the fly with probability 0.5
  using the specified `synth_dark` formula. No synthetic file count is invented.
- A group whose size differs from the specification raises
  `SplitSpecMismatch`. `--allow-count-mismatch` records the deviation instead;
  such a run is not the specified protocol.

## Commands

```bash
# Presence, expected counts, free disk space. Works with no data.
python -m training.data_pipeline check --sources

# Scan datasets and write datasets/manifests/manifest.jsonl.
python -m training.data_pipeline organise

# Probe every file with Pillow + OpenCV; report corruption/truncation.
python -m training.data_pipeline validate

# Deduplicate and detect cross-dataset leakage (pHash + SSIM), no writes.
python -m training.data_pipeline dedupe

# Write the count specification only (no data needed).
python -m training.data_pipeline split --spec-only

# Build the real split files once the datasets are present.
python -m training.data_pipeline split

# Generate 384x384 crops for one pool.
python -m training.data_pipeline preprocess --pool fine_tune --augment --darken

# Generate fixed synthetic dark images (count is mandatory).
python -m training.data_pipeline synthetic --sources datasets/raw/sice/part1 --count 100

# Inventory and cleaning report.
python -m training.data_pipeline report

# Everything, including split files and reports.
python -m training.data_pipeline all
```

Generated files:

- `datasets/manifests/manifest.jsonl` (git-ignored)
- `datasets/processed/<pool>/<dataset>/{low,high}/` (git-ignored)
- `training/splits/{fine_tune,validation,test,unseen,do_no_harm}.json` plus
  `split_spec.json` and `split_report.json` (committed)
- `training/reports/cleaning_report.{json,md}` (committed)

## Duplicate and leakage rules

1. 64-bit perceptual hash (`imagehash.phash`, 8×8) on every fine-tune,
   validation and test image.
2. Any train/validation image within Hamming distance 8 of any test image is a
   candidate.
3. Candidates are compared as 64×64 grayscale copies; SSIM above 0.9 means the
   train/validation image is dropped.
4. Duplicates inside the fine-tune pool use the same two-pass rule, keeping
   LOL-v1 over SICE over LOL-v2-Real for deterministic reruns.
5. Test images are never dropped.

The Hamming search uses a four-block pigeonhole index (radius 2) with exact
popcount verification, so it is exhaustive without comparing every pair.

## Sources

| Dataset | Source | Verified | Access |
| --- | --- | --- | --- |
| LOL-v1 | <https://daooshee.github.io/BMVC2018website/> | yes | Direct Google Drive / Baidu links on the project page |
| LOL-v2-Real | Authors' dataset page (not verified here) | no | Obtain from the paper/dataset page or a mirror you trust |
| LOL-v2-Synthetic | Authors' dataset page (not verified here) | no | Same as above |
| SICE | <https://github.com/csjcai/SICE> | yes | Download links in the README (Baidu / Google Drive) |
| DICM, LIME, MEF, NPE, VV | <https://github.com/baidut/BIMEF> | yes | Collection repository links a Google Drive archive |
| ExDark | <https://github.com/cs-chan/Exclusively-Dark-Image-Dataset> | yes | Direct download; BSD-3-Clause |
| Own captures | Your phone | — | 60 unpaired + about 20 paired scenes |

Run `python -m training.data_pipeline sources` for the full note on access,
licence and expected size for each dataset.

### Manual downloads required

1. LOL-v1
2. LOL-v2-Real
3. LOL-v2-Synthetic
4. SICE Part 1 (and the curated Part 2 low-light selection)
5. DICM, LIME, MEF, NPE, VV
6. ExDark
7. Your own 60 captures (+ 20 paired scenes if available)

No URL is invented. Unverified entries carry no download link, and no script
bypasses registration or licence terms.

## Guarantees

- Raw images are read-only; cleaning produces reports, not deletions.
- Counts and splits come from `final_decision.md`; deviations fail loudly.
- Train, validation, test and unseen pools are physically separate files and
  are asserted disjoint.
