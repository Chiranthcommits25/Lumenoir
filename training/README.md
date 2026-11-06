# Training and offline tooling

This folder holds everything that runs offline: dataset preparation, the
metric harness entry point and (later) remote training notebooks. None of it is
imported by the FastAPI application.

## Layout

```
training/
├── data_pipeline/            # dataset preparation package
│   ├── config.py             # constants + exact split counts
│   ├── sources.py            # official sources and licences
│   ├── organisation.py       # discovery and low/high pairing
│   ├── validation.py         # corruption / truncation detection
│   ├── hashing.py            # 64-bit pHash + exact Hamming index
│   ├── leakage.py            # duplicate + cross-dataset leakage
│   ├── preprocessing.py      # decode, resize, crop, augment
│   ├── synthetic.py          # specified synth_dark formula
│   ├── splits.py             # deterministic split generation
│   ├── statistics.py         # cleaning reports
│   ├── pipeline.py           # orchestration
│   └── cli.py                # command line entry point
├── kaggle_eval/              # Kaggle discovery, runner, notebook, samples
├── evaluate.py               # wrapper for backend.evaluation.cli
├── splits/                   # committed split spec / split files
└── reports/                  # committed cleaning reports
```

## Install

```bash
pip install -r requirements-eval.txt      # numpy, OpenCV, scikit-image, imagehash, SciPy, psutil
pip install -r requirements-metrics.txt   # optional: torch, torchvision, lpips, pyiqa
```

The API image only installs `requirements.txt`; the evaluation stack never
enters the deployed service.

## Dataset pipeline

```bash
python -m training.data_pipeline check --sources
python -m training.data_pipeline split --spec-only
python -m training.data_pipeline organise
python -m training.data_pipeline validate
python -m training.data_pipeline dedupe
python -m training.data_pipeline split
python -m training.data_pipeline preprocess --pool fine_tune --augment --darken
python -m training.data_pipeline synthetic --sources datasets/raw/sice/part1 --count 100
python -m training.data_pipeline report
python -m training.data_pipeline all
```

Raw images are read-only. Generated crops go to `datasets/processed/`; split
files and reports go to `training/splits/` and `training/reports/`.

Local data is present under `datasets/raw` (LOL-v1, LOL-v2, SICE_Paired and
extracted NPE/LIME/MEF/VV); ExDark/DICM are absent. See
[`../docs/datasets.md`](../docs/datasets.md) for the expected tree, the exact
counts, the sources to download manually and the cleaning rules.

## Evaluation

```bash
python -m training.evaluate --pairs pairs.json --output reports/eval
```

See [`../docs/evaluation.md`](../docs/evaluation.md) for the metric definitions,
aggregation and the integration-test split.

## Kaggle evaluation

```bash
python -m training.kaggle_eval --discover-only \
  --input-root /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets \
  --additional-root /kaggle/input/datasets/washingtongold/exdark-dataset \
  --additional-root /kaggle/input/datasets/twozeb/dicm-img \
  --additional-root /kaggle/working \
  --output-dir /kaggle/working/lumenoir-evaluation
python -m training.kaggle_eval --config /kaggle/working/eval_config.json
```

The runner reuses `backend.enhancement` and `backend.evaluation`. It writes
deterministic manifests, image-level records, aggregates, latency, failure logs,
device information and before/after visuals under `/kaggle/working/`. LIME,
MEF, NPE/NPE-ex1/2/3 and VV are unpaired; NPE and its extensions are grouped
into one `npe` report with subgroup metadata. Both train and test records are
discovered, but the runner validates manifests and evaluates only test/eval
records. See [`kaggle_eval/README.md`](kaggle_eval/README.md) for the explicit
dataset configuration and limitations.

## Training (not implemented)

Remote training notebooks and the fine-tuning configuration are still to be
written. Model architecture, losses and hyper-parameters are fixed in
`final_decision.md` §2 but no training has run. Trained weights are exported as
`.pth` files into `backend/weights/` (git-ignored) and never committed.
