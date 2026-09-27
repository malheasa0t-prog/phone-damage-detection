# Labeling with Label Studio

Bounding boxes for this project were drawn in
[Label Studio](https://labelstud.io/). These steps set it up in a separate
Anaconda environment so it does not conflict with the training dependencies.

```bash
conda create -n labelstudio python=3.10 -y
conda activate labelstudio
pip install --upgrade pip setuptools wheel
pip install label-studio
label-studio
```

The last command starts a local web server and opens Label Studio in the
browser. Create a project with an object-detection (bounding box) template and
add the six class names used by the model: Crack, Dents, Dislodged Screen,
Scratches, oil, phone good. Export the finished annotations in YOLO format to
use them with `scripts/train_yolo.py`.
