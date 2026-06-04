"""
Practice training a detector for RoboMaster convoy targets.

1. Collect images/video from HULA or mapping drone camera in arena-like conditions
2. Label with Roboflow / CVAT (class: robomaster)
3. Put dataset in datasets/robomaster/ with data.yaml
4. Run: python scripts/train_yolo.py

Export for mapping drone NPU: use organizer convertyolotoonnx.py + converttorknn.py on Discord.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "robomaster" / "data.yaml"


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return

    if not DATA_YAML.exists():
        print(f"Create dataset first: {DATA_YAML}")
        print("Example data.yaml:")
        print("  path: datasets/robomaster")
        print("  train: images/train")
        print("  val: images/val")
        print("  names: [robomaster]")
        return

    model = YOLO("yolov8n.pt")
    model.train(data=str(DATA_YAML), epochs=100, imgsz=640, project=str(ROOT / "runs"))
    best = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    if best.exists():
        dest = models_dir / "robomaster_best.pt"
        dest.write_bytes(best.read_bytes())
        print(f"Copied weights to {dest}")


if __name__ == "__main__":
    main()
