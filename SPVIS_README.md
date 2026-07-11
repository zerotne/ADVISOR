# SPVIS: Enhancing Video Instance Segmentation through Stabilized Feature Propagation

Yan Jin, Fang Gao, Qingjiao Meng, Qingbao Huang, Hanbo Zheng, and Shengheng Ma

[![Code](https://img.shields.io/badge/code-GitHub-181717.svg)](https://github.com/zerotne/SPVIS)
[![Release](https://img.shields.io/badge/release-Video--Instance--Segmentation-blue.svg)](https://github.com/zerotne/SPVIS/releases/tag/Video-Instance-Segmentation)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18326723.svg)](https://doi.org/10.5281/zenodo.18326723)

<!--
AUTHOR RELEASE CHECKLIST - DO NOT REMOVE UNTIL EVERY ITEM IS COMPLETE

1. Upload every checkpoint listed in the Model Zoo to a new version of the
   Zenodo record, replace every <SHA256> entry with the real checksum, and
   update the Zenodo creator/title/license metadata to match the final release.
2. Add environment.yml and requirements-lock.txt generated from the environment
   used for the final experiments.
3. Add a root LICENSE file and replace <SPVIS_LICENSE> below. Check compatibility
   with the licenses of Mask2Former, MinVIS, VITA, and DVIS-derived files.
4. Repair the released configuration files before publishing this README:
   - several _BASE_ entries refer to missing DVIS_*.yaml files;
   - META_ARCHITECTURE and decoder names use DVIS_* although the registered
     implementation classes are ADVISOR_*;
   - YouTube-VIS 2019/2021 registration is commented out in
     advisor/data_video/datasets/builtin.py;
   - the repository uses youtubevis_2023 paths where the paper reports
     YouTube-VIS 2022, and one test-json path contains a typo;
   - the YouTube-VIS 2021 offline Swin-L config currently points to 2019/2023
     datasets rather than the reported 2021 setting;
   - local absolute or malformed checkpoint paths must be removed.
5. Verify that the learning-rate schedule in the released configs matches the
   final paper and training logs.
6. Repair convert_coco2ytvis.py imports if the released package is advisor rather
   than vita.
7. Run every command in "Reproduce Table 1/Table 2/Table 3" from a clean clone.
8. Do not claim complete reproducibility in the response letter until the clean
   clone test, checkpoints, checksums, environment files, and license are public.
-->

Official implementation of SPVIS, a feature-propagation framework for video
instance segmentation. SPVIS combines a Progressive Tracker (PGT), a Refinement
Compensator (RCP), and a Spatial Interaction Module (SIM) to reduce temporal
association errors and preserve propagated object-query features.

![SPVIS architecture](utils/1.png)

## Contents

- [Reproducibility at a glance](#reproducibility-at-a-glance)
- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Model zoo](#model-zoo)
- [Training](#training)
- [Evaluation and inference](#evaluation-and-inference)
- [Reproduce Table 1](#reproduce-table-1)
- [Reproduce Table 2](#reproduce-table-2)
- [Reproduce Table 3](#reproduce-table-3)
- [Randomness and numerical variation](#randomness-and-numerical-variation)
- [Archive, license, and citation](#archive-license-and-citation)

## Reproducibility at a glance

| Item | Released specification |
|---|---|
| Code repository | <https://github.com/zerotne/SPVIS> |
| Permanent release tag | [`Video-Instance-Segmentation`](https://github.com/zerotne/SPVIS/releases/tag/Video-Instance-Segmentation) |
| Version-specific software DOI | [`10.5281/zenodo.18326724`](https://doi.org/10.5281/zenodo.18326724) |
| Concept DOI for the latest archived version | [`10.5281/zenodo.18326723`](https://doi.org/10.5281/zenodo.18326723) |
| Main entry point | `train_net_video.py` |
| Configuration files | `configs/` |
| Environment files | `environment.yml`, `requirements.txt`, and `requirements-lock.txt` |
| Default random seed | `42` (`advisor/config.py`) |
| Reported training hardware | 4 NVIDIA A100 GPUs |
| Reported global batch size | 8 |
| Optimizer and initial learning rate | AdamW, `1e-4` |
| Reported training length | 40,000 iterations |
| Training clip setup | 5 sampled clips per video, 3 frames per clip |
| Temporal window size | 12 |
| Training short-edge scales | 360, 480, and 600 pixels |
| PGT / RCP / SIM depth | 6 ECBs / 4 RCP blocks / 2 SIM blocks |
| Interaction boundary constant | `0.6` |

The configuration files are the executable source of truth for each released
model. Command-line overrides are written explicitly below so that the seed,
checkpoint, and output directory are recorded in every run.

## Repository layout

```text
SPVIS/
|-- advisor/                    # SPVIS models, modules, data loaders, evaluators
|-- configs/                    # dataset/backbone/online-offline configurations
|-- demo_video/                 # frame-directory visualization tools
|-- mask2former/                # Mask2Former components
|-- mask2former_video/          # video Mask2Former components
|-- utils/                      # evaluation and conversion utilities
|-- train_net_video.py          # training and evaluation entry point
|-- convert_coco2ytvis.py       # optional COCO category conversion
|-- environment.yml             # exact Conda environment for the release
|-- requirements.txt            # direct Python dependencies
|-- requirements-lock.txt       # complete package lock from the release environment
|-- LICENSE                     # SPVIS license and third-party notices
`-- README.md
```

Configuration filenames retain the historical `ADVISOR` prefix used during
implementation. They correspond to the SPVIS models reported in the paper.

## Installation

The reference setup uses Linux, Python 3.8, PyTorch 1.9.0, torchvision 0.10.0,
and CUDA 11.1.

### Option A: reproduce the archived environment

```bash
git clone https://github.com/zerotne/SPVIS.git
cd SPVIS

conda env create -f environment.yml
conda activate spvis
```

### Option B: install the reference environment manually

```bash
conda create -n spvis python=3.8 -y
conda activate spvis

pip install torch==1.9.0+cu111 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

pip install -r requirements.txt
pip install git+https://github.com/cocodataset/panopticapi.git
```

Compile the multi-scale deformable-attention CUDA operator:

```bash
export CUDA_HOME=/usr/local/cuda-11.1
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

Verify the installation:

```bash
python - <<'PY'
import torch
import detectron2
import advisor

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("SPVIS import: OK")
PY
```

For an exact audit of package versions, compare the active environment against
`requirements-lock.txt`.

## Dataset preparation

Download YouTube-VIS from the
[official YouTube-VOS VIS page](https://youtube-vos.org/dataset/vis/) and OVIS
from the [official OVIS page](https://songbai.site/ovis/). Dataset files are not
redistributed by this repository.

Set the dataset root before training or evaluation:

```bash
export DETECTRON2_DATASETS=/absolute/path/to/datasets
```

The released data loaders expect the following structure:

```text
$DETECTRON2_DATASETS/
|-- coco/
|   |-- train2017/
|   |-- val2017/
|   `-- annotations/
|       |-- instances_train2017.json
|       `-- instances_val2017.json
|-- ytvis_2019/
|   |-- train.json
|   |-- valid.json
|   |-- test.json
|   |-- train/JPEGImages/
|   |-- valid/JPEGImages/
|   `-- test/JPEGImages/
|-- ytvis_2021/
|   |-- train.json
|   |-- valid.json
|   |-- test.json
|   |-- train/JPEGImages/
|   |-- valid/JPEGImages/
|   `-- test/JPEGImages/
|-- ytvis_2022/
|   |-- train.json
|   |-- valid.json
|   |-- test.json
|   |-- train/JPEGImages/
|   |-- valid/JPEGImages/
|   `-- test/JPEGImages/
`-- ovis/
    |-- annotations/
    |   |-- annotations_train.json
    |   |-- annotations_valid.json
    |   `-- annotations_test.json
    |-- train/
    |-- valid/
    `-- test/
```

If COCO images are used for pseudo-video pretraining, generate the converted
annotation files after downloading COCO 2017:

```bash
python convert_coco2ytvis.py
```

This creates the category-filtered JSON files under
`$DETECTRON2_DATASETS/coco/annotations/`.

## Model zoo

All final checkpoints should be stored in a versioned Zenodo release under the
[SPVIS concept DOI](https://doi.org/10.5281/zenodo.18326723). Keep the exact
filenames below and publish a `checksums.sha256` file in the same archive.

### Table 1: YouTube-VIS 2019 and 2021 validation sets

| Dataset | Mode | Backbone | Config | AP | Checkpoint | SHA256 |
|---|---|---|---|---:|---|---|
| YouTube-VIS 2019 | Online | R50 | `configs/youtubevis_2019/ADVISOR_Online_R50.yaml` | 57.7 | `spvis_ytvis2019_online_r50.pth` | `<SHA256>` |
| YouTube-VIS 2019 | Online | Swin-L | `configs/youtubevis_2019/swin/ADVISOR_Online_SwinL.yaml` | 68.1 | `spvis_ytvis2019_online_swinl.pth` | `<SHA256>` |
| YouTube-VIS 2019 | Offline | R50 | `configs/youtubevis_2019/ADVISOR_Offline_R50.yaml` | 58.9 | `spvis_ytvis2019_offline_r50.pth` | `<SHA256>` |
| YouTube-VIS 2019 | Offline | Swin-L | `configs/youtubevis_2019/swin/ADVISOR_Offline_SwinL.yaml` | 69.5 | `spvis_ytvis2019_offline_swinl.pth` | `<SHA256>` |
| YouTube-VIS 2021 | Online | R50 | `configs/youtubevis_2021/ADVISOR_Online_R50.yaml` | 51.7 | `spvis_ytvis2021_online_r50.pth` | `<SHA256>` |
| YouTube-VIS 2021 | Online | Swin-L | `configs/youtubevis_2021/swin/ADVISOR_Online_SwinL.yaml` | 62.1 | `spvis_ytvis2021_online_swinl.pth` | `<SHA256>` |
| YouTube-VIS 2021 | Offline | R50 | `configs/youtubevis_2021/ADVISOR_Offline_R50.yaml` | 53.3 | `spvis_ytvis2021_offline_r50.pth` | `<SHA256>` |
| YouTube-VIS 2021 | Offline | Swin-L | `configs/youtubevis_2021/swin/ADVISOR_Offline_SwinL.yaml` | 64.6 | `spvis_ytvis2021_offline_swinl.pth` | `<SHA256>` |

### Table 2: OVIS validation set

| Mode | Backbone | Config | AP | Checkpoint | SHA256 |
|---|---|---|---:|---|---|
| Online | R50 | `configs/ovis/ADVISOR_Online_R50.yaml` | 39.8 | `spvis_ovis_online_r50.pth` | `<SHA256>` |
| Online | Swin-L | `configs/ovis/swin/ADVISOR_Online_SwinL.yaml` | 52.1 | `spvis_ovis_online_swinl.pth` | `<SHA256>` |
| Offline | R50 | `configs/ovis/ADVISOR_Offline_R50.yaml` | 41.1 | `spvis_ovis_offline_r50.pth` | `<SHA256>` |
| Offline | Swin-L | `configs/ovis/swin/ADVISOR_Offline_SwinL.yaml` | 54.3 | `spvis_ovis_offline_swinl.pth` | `<SHA256>` |

### Table 3: YouTube-VIS 2022 validation set

| Backbone | Config | AP | Checkpoint | SHA256 |
|---|---|---:|---|---|
| R50 | `configs/youtubevis_2022/ADVISOR_Offline_R50.yaml` | 41.3 | `spvis_ytvis2022_r50.pth` | `<SHA256>` |
| Swin-L | `configs/youtubevis_2022/swin/ADVISOR_Offline_SwinL.yaml` | 51.9 | `spvis_ytvis2022_swinl.pth` | `<SHA256>` |

Download the archived checkpoint bundle, place the files under `checkpoints/`,
and verify its checksum manifest:

```bash
sha256sum -c checkpoints/checksums.sha256
```

## Training

SPVIS uses a two-stage schedule. Train the online model first, then initialize
the offline model from the corresponding online checkpoint.

### Online training example

```bash
python train_net_video.py \
  --num-gpus 4 \
  --config-file configs/youtubevis_2019/ADVISOR_Online_R50.yaml \
  SEED 42 \
  OUTPUT_DIR outputs/train/ytvis2019_online_r50
```

### Offline training example

```bash
python train_net_video.py \
  --num-gpus 4 \
  --config-file configs/youtubevis_2019/ADVISOR_Offline_R50.yaml \
  MODEL.WEIGHTS outputs/train/ytvis2019_online_r50/model_final.pth \
  SEED 42 \
  OUTPUT_DIR outputs/train/ytvis2019_offline_r50
```

Resume an interrupted run with `--resume` and the same `OUTPUT_DIR`:

```bash
python train_net_video.py \
  --num-gpus 4 \
  --resume \
  --config-file configs/youtubevis_2019/ADVISOR_Online_R50.yaml \
  SEED 42 \
  OUTPUT_DIR outputs/train/ytvis2019_online_r50
```

Detectron2 writes the merged configuration and logs to `OUTPUT_DIR`. Preserve
these files with every released checkpoint.

## Evaluation and inference

Use `train_net_video.py --eval-only` for inference and official VIS evaluation:

```bash
python train_net_video.py \
  --num-gpus 1 \
  --eval-only \
  --config-file CONFIG.yaml \
  MODEL.WEIGHTS checkpoints/CHECKPOINT.pth \
  SEED 42 \
  OUTPUT_DIR outputs/eval/RUN_NAME
```

The evaluator writes raw predictions and YouTube-VIS-format results to:

```text
outputs/eval/RUN_NAME/inference/instances_predictions.pth
outputs/eval/RUN_NAME/inference/results.json
```

For validation splits, AP and AR are printed to the terminal and saved in the
Detectron2 log. For test splits without public annotations, submit
`inference/results.json` to the official evaluation protocol or evaluate against
the subsequently released official ground truth.

The following helper keeps all table-reproduction commands concise while still
recording the exact config, checkpoint, seed, and output path:

```bash
export NUM_GPUS=1
export WEIGHTS_DIR="$PWD/checkpoints"

run_eval () {
  config="$1"
  checkpoint="$2"
  output="$3"
  python train_net_video.py \
    --num-gpus "$NUM_GPUS" \
    --eval-only \
    --config-file "$config" \
    MODEL.WEIGHTS "$WEIGHTS_DIR/$checkpoint" \
    SEED 42 \
    OUTPUT_DIR "$output"
}
```

## Reproduce Table 1

Run all SPVIS rows reported for YouTube-VIS 2019 and 2021:

```bash
# YouTube-VIS 2019
run_eval configs/youtubevis_2019/ADVISOR_Online_R50.yaml \
  spvis_ytvis2019_online_r50.pth outputs/table1/ytvis2019_online_r50
run_eval configs/youtubevis_2019/swin/ADVISOR_Online_SwinL.yaml \
  spvis_ytvis2019_online_swinl.pth outputs/table1/ytvis2019_online_swinl
run_eval configs/youtubevis_2019/ADVISOR_Offline_R50.yaml \
  spvis_ytvis2019_offline_r50.pth outputs/table1/ytvis2019_offline_r50
run_eval configs/youtubevis_2019/swin/ADVISOR_Offline_SwinL.yaml \
  spvis_ytvis2019_offline_swinl.pth outputs/table1/ytvis2019_offline_swinl

# YouTube-VIS 2021
run_eval configs/youtubevis_2021/ADVISOR_Online_R50.yaml \
  spvis_ytvis2021_online_r50.pth outputs/table1/ytvis2021_online_r50
run_eval configs/youtubevis_2021/swin/ADVISOR_Online_SwinL.yaml \
  spvis_ytvis2021_online_swinl.pth outputs/table1/ytvis2021_online_swinl
run_eval configs/youtubevis_2021/ADVISOR_Offline_R50.yaml \
  spvis_ytvis2021_offline_r50.pth outputs/table1/ytvis2021_offline_r50
run_eval configs/youtubevis_2021/swin/ADVISOR_Offline_SwinL.yaml \
  spvis_ytvis2021_offline_swinl.pth outputs/table1/ytvis2021_offline_swinl
```

Expected SPVIS validation results:

| Dataset | Mode | Backbone | AP | AP50 | AP75 | AR1 | AR10 |
|---|---|---|---:|---:|---:|---:|---:|
| YouTube-VIS 2019 | Online | R50 | 57.7 | 80.4 | 63.1 | 51.8 | 64.3 |
| YouTube-VIS 2019 | Online | Swin-L | 68.1 | 89.7 | 75.0 | 59.7 | 73.4 |
| YouTube-VIS 2019 | Offline | R50 | 58.9 | 82.6 | 64.5 | 53.0 | 65.6 |
| YouTube-VIS 2019 | Offline | Swin-L | 69.5 | 92.9 | 77.8 | 61.7 | 75.9 |
| YouTube-VIS 2021 | Online | R50 | 51.7 | 75.7 | 55.1 | 44.3 | 59.8 |
| YouTube-VIS 2021 | Online | Swin-L | 62.1 | 84.5 | 69.9 | 48.6 | 67.9 |
| YouTube-VIS 2021 | Offline | R50 | 53.3 | 75.6 | 58.9 | 44.0 | 60.2 |
| YouTube-VIS 2021 | Offline | Swin-L | 64.6 | 86.8 | 72.1 | 49.6 | 70.5 |

## Reproduce Table 2

Run all SPVIS rows reported for OVIS:

```bash
run_eval configs/ovis/ADVISOR_Online_R50.yaml \
  spvis_ovis_online_r50.pth outputs/table2/ovis_online_r50
run_eval configs/ovis/swin/ADVISOR_Online_SwinL.yaml \
  spvis_ovis_online_swinl.pth outputs/table2/ovis_online_swinl
run_eval configs/ovis/ADVISOR_Offline_R50.yaml \
  spvis_ovis_offline_r50.pth outputs/table2/ovis_offline_r50
run_eval configs/ovis/swin/ADVISOR_Offline_SwinL.yaml \
  spvis_ovis_offline_swinl.pth outputs/table2/ovis_offline_swinl
```

Expected SPVIS validation results:

| Mode | Backbone | AP | AP50 | AP75 | AR1 | AR10 |
|---|---|---:|---:|---:|---:|---:|
| Online | R50 | 39.8 | 63.6 | 41.1 | 18.0 | 44.4 |
| Online | Swin-L | 52.1 | 75.5 | 57.2 | 20.9 | 57.0 |
| Offline | R50 | 41.1 | 64.2 | 42.8 | 18.1 | 46.7 |
| Offline | Swin-L | 54.3 | 78.9 | 59.3 | 20.9 | 59.9 |

## Reproduce Table 3

Run the reported YouTube-VIS 2022 configurations:

```bash
run_eval configs/youtubevis_2022/ADVISOR_Offline_R50.yaml \
  spvis_ytvis2022_r50.pth outputs/table3/ytvis2022_r50
run_eval configs/youtubevis_2022/swin/ADVISOR_Offline_SwinL.yaml \
  spvis_ytvis2022_swinl.pth outputs/table3/ytvis2022_swinl
```

Expected SPVIS validation results:

| Backbone | AP | AP50 | AP75 | AR1 | AR10 |
|---|---:|---:|---:|---:|---:|
| R50 | 41.3 | 60.6 | 44.1 | 36.7 | 47.9 |
| Swin-L | 51.9 | 73.2 | 54.7 | 41.5 | 56.6 |

## Randomness and numerical variation

The released default seed is `42`. Every command above passes `SEED 42`
explicitly. To run additional seeds, change only the seed and output directory:

```bash
for seed in 1 42 3407; do
  python train_net_video.py \
    --num-gpus 4 \
    --config-file configs/youtubevis_2019/ADVISOR_Online_R50.yaml \
    SEED "$seed" \
    OUTPUT_DIR "outputs/seeds/ytvis2019_online_r50_seed${seed}"
done
```

GPU kernels, distributed data loading, and library versions can introduce small
numerical differences. For a strict comparison, use the archived environment,
the stated checkpoint checksum, the same dataset annotations, and the same
number of GPUs. Report the exact git commit, config, seed, and output log with
every reproduced result.

## Archive, license, and citation

### Archival record

- Permanent GitHub tag:
  [`Video-Instance-Segmentation`](https://github.com/zerotne/SPVIS/releases/tag/Video-Instance-Segmentation)
- Version-specific Zenodo DOI:
  [`10.5281/zenodo.18326724`](https://doi.org/10.5281/zenodo.18326724)
- Concept DOI resolving to the latest archived release:
  [`10.5281/zenodo.18326723`](https://doi.org/10.5281/zenodo.18326723)

The archived release should contain source code, configuration files,
environment files, final checkpoints, and `checksums.sha256`.

### License

SPVIS-specific code is released under `<SPVIS_LICENSE>`; see [`LICENSE`](LICENSE).
Third-party components retain their original licenses and copyright notices.
In particular, users should consult the notices associated with Mask2Former,
MinVIS, VITA, DVIS, Detectron2, Swin Transformer, and the deformable-attention
operator. Dataset licenses are governed by their respective providers.

### Citation

```bibtex
@article{jin2026spvis,
  title   = {SPVIS: Enhancing Video Instance Segmentation through Stabilized Feature Propagation},
  author  = {Jin, Yan and Gao, Fang and Meng, Qingjiao and Huang, Qingbao and Zheng, Hanbo and Ma, Shengheng},
  journal = {The Visual Computer},
  year    = {2026}
}
```

For the software snapshot, cite the Zenodo record:

```bibtex
@software{jin2026spvis_code,
  author    = {Jin, Yan and Gao, Fang and Meng, Qingjiao and Huang, Qingbao and Zheng, Hanbo and Ma, Shengheng},
  title     = {SPVIS: Code and Reproducibility Materials},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18326724},
  url       = {https://doi.org/10.5281/zenodo.18326724}
}
```

## Acknowledgements

This implementation builds on
[Mask2Former](https://github.com/facebookresearch/Mask2Former),
[MinVIS](https://github.com/NVlabs/MinVIS),
[VITA](https://github.com/sukjunhwang/VITA), and
[DVIS](https://github.com/zhang-tao-whu/DVIS). We thank the authors for making
their work available.
