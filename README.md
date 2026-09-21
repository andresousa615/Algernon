# Algernon — selective facial anonymisation of head MRI

Algernon segments four facial structures (**ears, mouth, nose, eyes**) in 3D
head MRI with a MedNeXt network and anonymises each of them with a dedicated
transform. Because every region is a separate label, the user can choose
**which structures to anonymise and which to keep** (`--preserve_regions eyes`),
something classic defacing tools cannot offer.

The repository contains the full pipeline used in the dissertation:
pre-processing, distributed training (PyTorch DDP, FP16, `torch.compile`, a
work-stealing data loader), native-resolution inference and anonymisation,
and the privacy evaluation (face-detector defacing score and ArcFace
re-identification risk).

```
raw NIfTI/DICOM ─► RAS + resize 128³ ─► MedNeXt ─► argmax mask ─► restore to
native resolution ─► AnonymizationOfficer (per region) ─► anonymised NIfTI
```

Reference numbers are in [`docs/results/RESULTS.md`](docs/results/RESULTS.md):
Dice 0.754 on 96 external exams at native resolution, 52 % of anonymised
renders with no detectable face, rank-1 re-identification 4.3 % (chance 1 %).

---

## Repository layout

| Path | Content |
|---|---|
| `train.py` | DDP training (AMP, `torch.compile`, warm-up + cosine LR, snapshots, optional profiling) |
| `inference.py` | DDP inference, Dice at network and native resolution, anonymisation, timing/MFU |
| `configs/train_128.yaml` | Model, input size, paths and hyper-parameters |
| `data/dataset.py` | Datasets: RAS reorientation, DICOM → NIfTI, TorchIO augmentation, label remap |
| `data/minato_loader.py` | `MinatoSegLoader`: out-of-order, work-stealing loader with a background pin thread |
| `models/` | `MedNeXtSeg` wrapper and the vendored MedNeXt architecture |
| `anonymization/officer.py` | `AnonymizationOfficer`: per-region covers and fills, preservation logic |
| `preprocessing/` | `resample_to_128.py`, `make_splits.py`, `build_external_csv.py` |
| `evaluation/` | 3D renders (PyVista), defacing score (dlib), re-identification (ArcFace) |
| `tools/` | Training curves, per-class Dice, result summaries, preservation diagnostics |
| `profiling/` | `TrainingProfiler`, MFU tracker, NVTX helpers, nsys launcher, loader simulation |
| `slurm/` | SLURM templates for the whole pipeline and for each stage on its own |

## Installation

```bash
conda create -n algernon python=3.10
conda activate algernon
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# optional, for evaluation/ (renders, defacing score, re-identification):
pip install -r requirements-eval.txt        # see evaluation/README.md
```

Tested with Python 3.10, PyTorch 2.5.1 + CUDA 12.1 on NVIDIA A100-40GB.

## Data format

One folder per exam. Ground-truth masks use labels `{0, 2, 3, 4, 5}` for
`{background, ears, mouth, nose, eyes}` and are remapped to `{0..4}` on load.

```
dataset/
└── <subject>__<session>/
    ├── raw.nii.gz
    └── training_masks/mask_4_classes.nii.gz      # training/validation/test only
```

The subject id is the folder name up to the first `__`; splits are made per
subject so that no subject appears in two sets. Exams for inference only need
`raw.nii.gz` (or a DICOM series).

> The datasets used in the dissertation (ADNI, IXI) are distributed under
> their own data-use agreements and are **not** included. Ground-truth masks
> were produced with a separate tool and are not part of this repository.

## Workflow

### 1. Pre-processing

```bash
# resample every exam (and its mask, if present) to the network input size
python preprocessing/resample_to_128.py /path/dataset /path/dataset_128

# subject-level 60/20/20 split -> data/train.csv, val.csv, test.csv
python preprocessing/make_splits.py --root /path/dataset_128 --out data

# CSV for an external set without ground truth
python preprocessing/build_external_csv.py --root /path/external --out data/external.csv
```

Each CSV has the columns `image_path,mask_path` (see `data/example.csv`);
`mask_path` may be empty for inference-only sets.

### 2. Training

```bash
torchrun --nproc_per_node=2 train.py --config configs/train_128.yaml --epochs 100
```

Multi-node launches go through `slurm/pipeline_train_eval.sh`. Outputs land in
`results/train_<model>_<lr>_<comment>_<job_id>/`: `best_*.pt`,
`training_metrics.csv`, a copy of the config and, with `--profile`, the
profiler and MFU reports. A snapshot is written every epoch and can be
resumed with `--resume_id <job_id>`.

Training details: batch size 1 per GPU, AdamW `lr=5e-4` (double it for every
doubling of the global batch), 5-epoch linear warm-up then cosine annealing,
Dice + 0.1·CE loss, FP16 autocast, `torch.compile(mode="max-autotune-no-cudagraphs")`.
Augmentation: one spatial transform per sample (affine 75 % / elastic 25 %,
scheduled by the loader) and bias field / noise / gamma / spike each with p = 0.25.

### 3. Inference and anonymisation

```bash
torchrun --nproc_per_node=2 inference.py \
    --config configs/train_128.yaml \
    --weights results/<run>/best_mednext_0.0005_mednext_128.pt \
    --test_csv data/test.csv data/external.csv \
    --out_dir results/<run>/inference \
    --metrics_csv results/<run>/test_metrics.csv \
    --metrics_txt results/<run>/test_metrics.txt \
    --preserve_regions eyes            # or --anonymise_regions nose,mouth
```

For every exam this writes `<exam>_pred_mask.nii.gz` and `<exam>_anon.nii.gz`
at the exam's native resolution and orientation, plus per-exam Dice (when a
mask exists) and timing summaries.

Region transforms (`anonymization/officer.py`):

| Region | Cover | Fill |
|---|---|---|
| nose | mask dilated ×2 | zero |
| eyes | mask + 12-voxel anterior cylindrical projection | plateau (max of smoothed volume inside the eyes) |
| ears | mask dilated ×6 laterally, clipped medially per axial slice | random noise below the air threshold |
| mouth | mask | plateau (max of smoothed volume inside the mouth) |

When some regions are preserved, the cover of each preserved region is
restored from the original volume at the end; where it overlaps the cover of
an anonymised region, anonymisation wins.

### 4. Privacy evaluation (optional)

```bash
# 2D renders of original/anonymised pairs (PyVista; xvfb-run on headless nodes)
xvfb-run -a python evaluation/render_pairs_ddp.py \
    --dir_defaced results/<run>/inference --test_csvs data/test.csv --dir_pairs results/<run>/pairs_2d

# defacing score: fraction of pairs whose anonymised render has no detectable face
python evaluation/defacing_score_ddp.py --dir_pairs results/<run>/pairs_2d --output_report results/<run>/defacing_report.txt

# re-identification risk with ArcFace (separate conda env, see evaluation/README.md)
python evaluation/reidentification.py --pairs_dir results/<run>/pairs_2d --out results/<run>/reidentification.txt
```

### 5. Profiling (optional)

`train.py --profile` enables the `TrainingProfiler` (per-phase time and VRAM,
NCCL overhead, loader starvation, Chrome trace with `--profile-epochs N`) and
the MFU tracker. `slurm/pipeline_nsys_profile.sh` runs a short training under
Nsight Systems with one `nsys` process per rank.
`profiling/simulate_loader.py` is an analytical model of the loader strategies.

## The work-stealing loader

`torch.utils.data.DataLoader` assigns indices to workers round-robin and
delivers batches in order, so one slow sample (an elastic deformation costs
~4× an affine one) stalls the GPU even when other workers have batches ready.
`MinatoSegLoader` keeps a shared work queue that workers pull from, delivers
samples as they complete, pins batches in a background thread and
pre-schedules the expensive transforms 3:1 so they never cluster on one
worker. It is a drop-in replacement for the loader in `train.py`; see the
module docstring for the protocol.

## Pretrained weights

Weights are not distributed with the repository yet. A download link will be
added here once the data-use conditions of the training sets are settled.

## Citation

If you use this code, please cite the dissertation:

```
@mastersthesis{sousa2026algernon,
  author = {André Sousa},
  title  = {Algernon: selective facial anonymisation of head MRI with distributed deep learning},
  school = {<university>},
  year   = {2026}
}
```

## License

The original code in this repository is released under the Apache License 2.0
(`LICENSE`). Two third-party components are included with their own terms —
see `NOTICE`:

* **MedNeXt** (DKFZ), Apache 2.0 — `models/mednext/`.
* **mede** (Heine & Rempe), Apache 2.0 **with the Commons Clause** — the
  training-loop, dataset and utility skeletons in `train.py`, `data/dataset.py`,
  `models/mednext_seg.py` and `utils/` started as modified copies of it. The
  Commons Clause restriction on commercial exploitation continues to apply to
  those portions.
