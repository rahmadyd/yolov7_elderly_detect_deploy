import csv
from datetime import datetime
from pathlib import Path


CSV_COLUMNS = [
    "timestamp",
    "filename",
    "class_name",
    "confidence",
    "x",
    "y",
    "width",
    "height",
]

HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_FILE = HISTORY_DIR / "detections.csv"


def save_detection_history(filename, detections):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_FILE.exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        if not detections:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "filename": filename,
                    "class_name": "none",
                    "confidence": 0,
                    "x": "",
                    "y": "",
                    "width": "",
                    "height": "",
                }
            )
            return

        for detection in detections:
            x, y, width, height = detection["box"]
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "filename": filename,
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )


def read_detection_history():
    if not HISTORY_FILE.exists():
        return []

    with HISTORY_FILE.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
