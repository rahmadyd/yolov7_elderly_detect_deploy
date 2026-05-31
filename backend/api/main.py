from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detector import YOLOv7ONNXDetector
from history import read_detection_history, save_detection_history


app = FastAPI(title="YOLOv7 Elderly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = YOLOv7ONNXDetector()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "YOLOv7 Elderly Detection API is running"}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        image_bytes = await file.read()
        detections = detector.detect(image_bytes)
        save_detection_history(file.filename, detections)

        return {
            "status": "success",
            "filename": file.filename,
            "detections": detections,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/history")
def get_history():
    return {"status": "success", "history": read_detection_history()}
