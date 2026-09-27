# Phone Damage Detection

YOLO-based computer vision project for detecting and classifying visible phone
damage from images. The repository includes reusable utilities for image
augmentation, duplicate-image cleanup, YOLO training, inference, and saved
training artifacts from earlier experiments.

## Project Goals

- Detect common phone-condition classes from images.
- Support repeatable YOLO model training and inference.
- Keep preprocessing and evaluation scripts reusable outside the original notebook environment.

## Example Predictions

![Validation predictions from the 60-epoch model](models/yolo11s_60_epochs/val_batch0_pred.jpg)

Predictions of the 60-epoch model on a validation batch (confidence shown next to each box).

## Results

Three YOLO11s runs were trained. The best one is `models/yolo11s_60_epochs`
(best epoch 47 of 60), with these validation scores:

- Precision: 0.534
- Recall: 0.598
- mAP50: 0.551
- mAP50-95: 0.302

Performance is moderate. Oil, Scratches and Phone Good score best,
while Crack is often missed and Dents almost always fail. See
[RESULTS.md](RESULTS.md) for all runs, per-class scores, plots, and next steps.

## Classes

The dataset uses six phone-condition classes:

- Crack
- Dents
- Dislodged Screen
- Scratches
- Oil
- Phone Good

`scripts/run_inference.py` reads the class names stored in the trained
weights, so predicted labels always match the training order. The model in
`models/yolo11s_260_epochs` was trained without Dislodged Screen and has five
classes.

## Repository Structure

```text
scripts/          Reusable Python utilities
models/           Trained weights and YOLO training outputs for each run
docs/             Setup notes (labeling with Label Studio)
RESULTS.md        Metrics, analysis, and next steps
TEST.ipynb        Small notebook used for a first inference test
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Augment Images

```bash
python scripts/augment_images.py --input-dir path/to/images --copies 3
```

### Remove Duplicate Images

Run a dry check first:

```bash
python scripts/remove_duplicate_images.py --image-dir path/to/images
```

Delete near-duplicates after reviewing the output:

```bash
python scripts/remove_duplicate_images.py --image-dir path/to/images --delete
```

### Train YOLO

```bash
python scripts/train_yolo.py --data data.yaml --model yolo11s.pt --epochs 60 --imgsz 640
```

### Run Inference

```bash
python scripts/run_inference.py --model models/yolo11s_60_epochs/weights/best.pt --source path/to/test.jpg
```

### Summarize a Training Run

```bash
python scripts/summarize_results.py models/yolo11s_60_epochs/results.csv
```

### Label New Images

See [docs/labeling-with-label-studio.md](docs/labeling-with-label-studio.md).

## Notes

- Model weights are stored directly in the repository. The 160- and 250-epoch
  run folders also contain a `my_model.zip` archive with the same weights and
  plots.
- The training dataset is not included in this repository.
