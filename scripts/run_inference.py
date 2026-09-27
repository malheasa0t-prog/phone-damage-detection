"""Run YOLO inference for phone damage detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phone damage inference.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="phone_damage")
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    # Class names are read from the trained weights, so labels always match
    # the class order used during training.
    model = YOLO(str(args.model))

    results = model(
        str(args.source),
        save=args.save,
        project=args.project,
        name=args.name,
    )

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model.names[class_id]
            print(f"{label} {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")


if __name__ == "__main__":
    main()
