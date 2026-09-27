# Training Results

This page summarizes the three YOLO11s training runs saved under `models/`.
All numbers come from each run's `results.csv`, `args.yaml`, and the plots
that Ultralytics saved at the end of training. Metrics are on the validation
split, IoU-based, for bounding-box detection.

To reproduce the tables below:

```bash
python scripts/summarize_results.py models/yolo11s_60_epochs/results.csv
python scripts/summarize_results.py models/yolo11s_160_epochs/train/results.csv
python scripts/summarize_results.py models/yolo11s_260_epochs/train/results.csv
```

## Training setup

All three runs used the same settings in `args.yaml`: pretrained `yolo11s.pt`,
image size 640, batch size 16, `optimizer: auto`, `lr0: 0.01`, default
Ultralytics augmentation (mosaic 1.0, horizontal flip 0.5), and training on
Google Colab (`data: /content/data.yaml`).

The dataset was not identical across runs. The `labels.jpg` plot of each run
shows these training-label counts:

| Class            | 60-epoch run | 160-epoch run | 250-epoch run |
|------------------|-------------:|--------------:|--------------:|
| Crack            | 281          | 291           | 214           |
| Scratches        | 183          | 173           | 144           |
| Dents            | 91           | 89            | 46            |
| Phone Good       | 55           | 54            | 36            |
| Oil              | 38           | 45            | 45            |
| Dislodged Screen | 25           | 25            | not included  |

The folder `yolo11s_260_epochs` was actually trained for 250 epochs
(`epochs: 250` in `args.yaml`, 250 rows in `results.csv`), and its model has
only 5 classes. Because the data differs, the runs are not a strict
like-for-like comparison.

## Metrics per run

The best epoch is the one with the highest mAP50-95.

| Run folder            | Epochs | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|-----------------------|-------:|-----------:|----------:|-------:|------:|---------:|
| `yolo11s_60_epochs`   | 60     | 47         | 0.534     | 0.598  | 0.551 | 0.302    |
| `yolo11s_160_epochs`  | 160    | 133        | 0.726     | 0.436  | 0.440 | 0.253    |
| `yolo11s_260_epochs`  | 250    | 220        | 0.753     | 0.494  | 0.526 | 0.298    |

Final-epoch values, for reference:

| Run folder            | Final epoch | Precision | Recall | mAP50 | mAP50-95 |
|-----------------------|------------:|----------:|-------:|------:|---------:|
| `yolo11s_60_epochs`   | 60          | 0.718     | 0.517  | 0.546 | 0.266    |
| `yolo11s_160_epochs`  | 160         | 0.740     | 0.432  | 0.403 | 0.225    |
| `yolo11s_260_epochs`  | 250         | 0.672     | 0.431  | 0.446 | 0.259    |

## Best run: `yolo11s_60_epochs`

The 60-epoch run has the highest mAP50-95 (0.302 at epoch 47) and also the
highest mAP50 of any epoch in any run (0.595 at epoch 41). It is also the
shortest run (about 10 minutes of training time according to `results.csv`).
The 250-epoch run comes close (0.298), but it was trained on 5 classes only.

Per-class mAP50 for this run, from `BoxPR_curve.png`:

| Class            | mAP50 |
|------------------|------:|
| Oil              | 0.939 |
| Scratches        | 0.753 |
| Phone Good       | 0.708 |
| Crack            | 0.353 |
| Dents            | 0.003 |
| Dislodged Screen | no validation instances |
| All classes      | 0.551 |

The best overall F1 score is 0.55, reached at a confidence threshold of about
0.20 (`BoxF1_curve.png`).

Plots for this run:

![Precision-recall curve](models/yolo11s_60_epochs/BoxPR_curve.png)

![Confusion matrix](models/yolo11s_60_epochs/confusion_matrix.png)

Other plots in the same folder:
[F1 curve](models/yolo11s_60_epochs/BoxF1_curve.png),
[precision curve](models/yolo11s_60_epochs/BoxP_curve.png),
[recall curve](models/yolo11s_60_epochs/BoxR_curve.png),
[normalized confusion matrix](models/yolo11s_60_epochs/confusion_matrix_normalized.png),
[training curves](models/yolo11s_60_epochs/results.png),
[label distribution](models/yolo11s_60_epochs/labels.jpg).

## Analysis

Overall performance is moderate. A mAP50-95 around 0.30 and a mAP50 around
0.55 mean the model finds some damage reliably but is not ready for use
without a human check.

What the saved files show:

- **Oil is the strongest class in every run.** Its mAP50 is 0.939 in the
  60-epoch run and 0.995 in both longer runs, although it is scored on only
  10 validation objects in the 60-epoch run.
- **Dents almost always fail.** Dents mAP50 is 0.003 (60 epochs), 0.000
  (160 epochs) and 0.222 (250 epochs). In the 60-epoch run there are only 2
  Dents instances in the validation set, and both were missed.
- **Crack is often missed.** In the 60-epoch confusion matrix, 17 of 41
  validation cracks were detected and 24 were predicted as background. The
  model also predicted Crack on 11 background regions.
- **Dislodged Screen is never evaluated.** It has only 25 training labels and
  does not appear in the validation set of any run, so its quality is unknown.
- **The classes are imbalanced.** Crack has 281 training labels and Dislodged
  Screen has 25 (about 11 times fewer) in the 60-epoch run.
- **Longer training did not help.** In every run the training class loss kept
  falling (to 1.09, 0.56 and 0.46), while the validation class loss reached
  its lowest point early (epochs 46, 60 and 93) and then rose. This is a sign
  of overfitting on a small dataset.
- **The validation set is small.** The 60-epoch confusion matrix contains 78
  labeled objects in total, so one or two predictions can move a class score
  a lot. Per-class numbers should be read with care.

An earlier draft of these notes reported an overall F1 of 0.46. That value
matches the 160-epoch run (`models/yolo11s_160_epochs/train/BoxF1_curve.png`,
F1 0.46 at confidence 0.434), not the best run.

## Next steps

1. Collect and label more images for Dents and Dislodged Screen, and make sure
   every class appears in the validation split.
2. Use a fixed train/validation/test split so runs can be compared directly.
3. Train for fewer epochs, or keep early stopping on (`scripts/train_yolo.py`
   uses `patience=10`), since the best epochs came well before the end.
4. Try class-balanced sampling or targeted augmentation for rare classes
   (`scripts/augment_images.py`).
5. Review Crack labels and background images, since most Crack errors are
   misses or false alarms against the background.
6. Report results on a held-out test set that is never used during training.
