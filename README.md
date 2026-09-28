# Power-cable instance segmentation with RF-DETR

Instance segmentation of **power cables** in aerial images with [RF-DETR](https://github.com/roboflow/rf-detr) (`RFDETRSegPreview`). The repository contains the training scripts, inference, and an evaluation that, besides the COCO metrics, measures how well the model estimates the **orientation** of the cables.

## How it works

- **Model**: RF-DETR Seg with a single class (the cable), 100 queries and 60–80 predictions per image.
- **Two-stage training**:
  - [`train_epoch_0_to_11.py`](src/experiments/rf_detr/train_epoch_0_to_11.py): resolution 768 on the full dataset;
  - [`train_epoch_11_to_45.py`](src/experiments/rf_detr/train_epoch_11_to_45.py): resumes from an earlier checkpoint (`checkpoint0008.pth`) and continues at resolution 960 on a **tiled** version of the dataset, with different weights for the class, box and mask losses.
- **Evaluation** ([`inference.py`](src/experiments/rf_detr/inference.py)): for each checkpoint in `checkpoints/` it computes on the test set
  - AP@50 and AR@50:95 on masks and boxes (COCO);
  - an **angle score**: for each predicted cable matched to a ground-truth cable with IoU ≥ 0.3, the line of the mask is estimated (linear regression on its points) and its angle is compared with the ground truth, with similarity `exp(-0.12·Δθ)`;
  - the **LDS** (Line Detection Score) = AP@50 + AR@50:95 + 2 · angle score.

  Checkpoints are ranked by LDS in a CSV.
- [`run_lds_eval.py`](src/experiments/utils/run_lds_eval.py): computes AP@50, AR@50, angle score, rho difference and LDS from a predictions file in COCO format.
- [`rfdetr_seg_inference.py`](src/experiments/rf_detr/rfdetr_seg_inference.py): runs the model on a folder of images and saves the images with the predicted masks overlaid.

## Data

The data is not in the repository. The scripts expect a COCO/Roboflow-format dataset (`train`, `valid` and `test` folders, each with `_annotations.coco.json`) in:

| Path | Use |
|---|---|
| `data/rf-detr_data` | Full dataset (stage 1 and test) |
| `data/rf-detr_data_tiled` | Tiled dataset (stage 2) |

## Installation

A CUDA GPU is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the scripts from the repository root.

```bash
# stage 1
python src/experiments/rf_detr/train_epoch_0_to_11.py

# stage 2 (from an earlier checkpoint, see `weights=` in the script)
python src/experiments/rf_detr/train_epoch_11_to_45.py

# compare checkpoints on the test set (writes output/metrics_comparison.csv)
python src/experiments/rf_detr/inference.py

# LDS of a predictions file
python src/experiments/utils/run_lds_eval.py data/test/test.json predictions.json
```
