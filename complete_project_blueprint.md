# Panduan Lengkap: YOLOv7 Elderly Detection Web App
## Khusus Optimasi ASUS ROG Strix G531GD (Windows, GTX 1050 4GB)

Dokumen ini menggabungkan arsitektur, struktur folder, kode implementasi backend & frontend, serta panduan optimasi hardware agar aplikasi berjalan lancar (deteksi cepat & tidak lag) di laptop **ASUS ROG Strix G531GD**.

---

## 🚀 1. Strategi Optimasi Hardware (ROG Strix G531GD)

Laptop ROG Strix G531GD Anda dilengkapi dengan **NVIDIA GeForce GTX 1050 (4GB VRAM)** dan **Intel Core Gen 9**. Berikut cara memaksimalkannya:

### A. Gunakan Akselerasi GPU dengan DirectML
Biasanya untuk menjalankan model AI di GPU NVIDIA, kita harus menginstal CUDA Toolkit dan cuDNN secara manual yang ukurannya sangat besar (> 3 GB) dan versinya sensitif terhadap error.
*   **Solusi Ringan & Praktis**: Kita gunakan **DirectML (DirectX Machine Learning)** dari Microsoft.
*   DirectML memungkinkan FastAPI menggunakan kartu grafis GTX 1050 Anda secara otomatis melalui sistem Windows DirectX tanpa perlu instalasi CUDA/cuDNN yang rumit.
*   Library Python yang di-install cukup `onnxruntime-directml`.

### B. Batasi Penggunaan Thread CPU (jika tidak memakai GPU)
Jika Anda memilih menjalankan model di CPU, batasi jumlah thread maksimum menjadi **2 atau 4 thread**. Secara default, ONNX Runtime akan memakai seluruh core CPU (12 thread pada i7-9750H) yang membuat laptop Anda berisik (fan berputar kencang) dan lag saat membuka Chrome/VS Code secara bersamaan.

### C. Pemrosesan Video dengan Frame Skipping
Untuk deteksi video/webcam, jangan kirim 30 frame per detik (30 FPS) ke backend. Cukup kirim **6 hingga 10 frame per detik** (lewati 3 frame tiap memproses 1 frame). Secara visual, deteksi manusia masih terlihat real-time, tetapi beban kerja laptop Anda berkurang hingga 70%!

---

## 📐 2. Arsitektur System & Alur Request-Response

```
┌─────────────────────────────────┐        HTTP POST (Image File)        ┌───────────────────────────────┐
│       React JS (Frontend)       │ ───────────────────────────────────> │       FastAPI (Backend)       │
│                                 │                                      │                               │
│ 1. User upload gambar/webcam    │ <─────────────────────────────────── │ 1. Terima gambar              │
│ 2. Gambar asli dimuat ke layar │        JSON Data Koordinat Box       │ 2. Preprocess (Resize 640x640)│
│ 3. Canvas menggambar Box di     │                                      │ 3. Run Inference (DirectML)   │
│    atas gambar asli (Sangat     │                                      │ 4. Postprocess & Scale Box    │
│    ringan & responsif)          │                                      │ 5. Kirim data JSON koordinat  │
└─────────────────────────────────┘                                      └───────────────────────────────┘
```

---

## 📂 3. Struktur Folder Project

Buat folder project dengan struktur clean & modular seperti di bawah ini:

```txt
yolov7-elderly-detect/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # Entrypoint FastAPI & CORS
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py         # Logic ONNX Inference (DirectML/CPU)
│   │   │   └── utils.py            # Helpers preprocessing & postprocessing
│   │   │
│   │   └── weights/
│   │       └── yolov7-elderly.onnx # File model ONNX hasil convert
│   │
│   └── requirements.txt            # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── Header.jsx          # Navigasi atas yang clean
    │   │   ├── UploadArea.jsx      # Area drop gambar
    │   │   └── DetectionCanvas.jsx # Komponen Canvas untuk render box
    │   │
    │   ├── hooks/
    │   │   └── useDetection.js     # State upload & API Fetching
    │   │
    │   ├── App.jsx                 # Halaman utama
    │   ├── main.jsx
    │   └── index.css               # Setup Tailwind CSS
    │
    ├── package.json
    ├── tailwind.config.js
    └── vite.config.js
```

---

## 🛠️ 4. Instalasi & Setup Backend

### File: `backend/requirements.txt`
```text
fastapi==0.110.0
uvicorn==0.28.0
python-multipart==0.0.9
onnxruntime-directml==1.17.1
opencv-python-headless==4.9.0.80
numpy==1.26.4
pillow==10.2.0
```

> **INFO**: Kita menggunakan `onnxruntime-directml` untuk akselerasi GPU GTX 1050 di Windows. Jika dijalankan di Linux/macOS nanti, Anda cukup mengubahnya menjadi `onnxruntime` biasa di file requirements.

### File: `backend/app/services/detector.py`
```python
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

class YOLOv7Detector:
    def __init__(self, model_path: str):
        self.opts = ort.SessionOptions()
        # Batasi resource CPU agar laptop tidak hang
        self.opts.intra_op_num_threads = 2
        self.opts.inter_op_num_threads = 2
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # PENTING: Gunakan DirectML untuk akselerasi GPU di Windows (GTX 1050)
        # Jika DirectML gagal, sistem otomatis fallback ke CPU.
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        print(f"Loading ONNX Model dengan provider: {providers}")
        self.session = ort.InferenceSession(model_path, self.opts, providers=providers)
        
        # Metadata Model
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2] # 640
        self.input_width = self.input_shape[3]  # 640
        
        # Nama Kelas Deteksi Custom Anda (Sesuaikan!)
        self.classes = ["elderly", "fall", "walking"]

    def preprocess(self, image: Image.Image):
        # 1. Pastikan format gambar RGB
        img_np = np.array(image.convert("RGB"))
        orig_h, orig_w = img_np.shape[:2]
        
        # 2. Resize ke input size model (640x640)
        img_resized = cv2.resize(img_np, (self.input_width, self.input_height))
        
        # 3. Transpose HWC -> CHW
        img_transposed = img_resized.transpose(2, 0, 1)
        
        # 4. Normalisasi nilai pixel ke [0.0, 1.0]
        img_normalized = img_transposed.astype(np.float32) / 255.0
        
        # 5. Tambah dimensi batch (1, 3, 640, 640)
        input_tensor = np.expand_dims(img_normalized, axis=0)
        
        return input_tensor, orig_w, orig_h

    def detect(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes))
        input_tensor, orig_w, orig_h = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess (Parsing Bounding Box)
        return self.postprocess(outputs, orig_w, orig_h)

    def postprocess(self, outputs, orig_w, orig_h):
        predictions = outputs[0]
        detections = []
        
        for pred in predictions[0]:
            # Mendapatkan confidence score
            score = float(pred[6]) if len(pred) > 6 else float(pred[4])
            
            # Filter deteksi di bawah threshold 0.25 (25%)
            if score < 0.25:
                continue
                
            class_id = int(pred[5])
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            
            # Koordinat Bounding Box dari model ONNX (biasanya bernilai 0-640)
            # Skalakan kembali koordinat ke resolusi asli gambar
            x1 = float(pred[1]) / self.input_width * orig_w
            y1 = float(pred[2]) / self.input_height * orig_h
            x2 = float(pred[3]) / self.input_width * orig_w
            y2 = float(pred[4]) / self.input_height * orig_h
            
            w = x2 - x1
            h = y2 - y1
            
            detections.append({
                "box": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "confidence": round(score, 2),
                "class_id": class_id,
                "class_name": class_name
            })
            
        return detections
```

### File: `backend/app/main.py`
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.detector import YOLOv7Detector
import os

app = FastAPI(title="YOLOv7 Elderly Detection App", version="1.0.0")

# Izinkan komunikasi lintas port (CORS) antara Frontend (5173) dan Backend (8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model secara dinamis saat server mulai menyala
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "yolov7-elderly.onnx")
detector = None

@app.on_event("startup")
def load_model():
    global detector
    # Membuat mock detector jika model asli belum diletakkan di folder weights
    if not os.path.exists(WEIGHTS_PATH):
        print("⚠️ PERINGATAN: Model weights tidak ditemukan! Mengaktifkan MOCK DETECTOR untuk testing.")
        class MockDetector:
            def detect(self, contents):
                # Deteksi bohongan untuk testing UI React
                return [
                    {"box": [50, 60, 200, 300], "confidence": 0.89, "class_id": 0, "class_name": "elderly"},
                    {"box": [300, 400, 150, 100], "confidence": 0.92, "class_id": 1, "class_name": "fall"}
                ]
        detector = MockDetector()
    else:
        detector = YOLOv7Detector(WEIGHTS_PATH)

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    try:
        contents = await file.read()
        results = detector.detect(contents)
        return {"status": "success", "detections": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 💻 5. Instalasi & Setup Frontend (React)

### File: `frontend/src/hooks/useDetection.js`
```javascript
import { useState } from 'react';

export const useDetection = () => {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const uploadAndDetect = async (file) => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/detect/image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Gagal menghubungi API Backend.');
      }

      const data = await response.json();
      if (data.status === 'success') {
        setDetections(data.detections);
      } else {
        throw new Error('Gagal mendeteksi gambar.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setPreviewUrl(null);
    setDetections([]);
    setError(null);
  };

  return { previewUrl, detections, loading, error, uploadAndDetect, reset };
};
```

### File: `frontend/src/components/DetectionCanvas.jsx`
```jsx
import React, { useRef, useEffect } from 'react';

const DetectionCanvas = ({ previewUrl, detections }) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    if (!previewUrl) return;

    const img = new Image();
    img.src = previewUrl;
    imageRef.current = img;

    img.onload = () => draw();
  }, [previewUrl, detections]);

  useEffect(() => {
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [detections]);

  const draw = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    const displayWidth = canvas.parentElement.clientWidth;
    const scale = displayWidth / img.naturalWidth;
    const displayHeight = img.naturalHeight * scale;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

    detections.forEach((det) => {
      const [x, y, w, h] = det.box;

      // Skala ulang koordinat dari gambar asli ke ukuran display canvas
      const rx = x * scale;
      const ry = y * scale;
      const rw = w * scale;
      const rh = h * scale;

      let color = '#3b82f6'; // Biru untuk elderly/biasa
      if (det.class_name.toLowerCase() === 'fall') {
        color = '#ef4444'; // Merah jika terdeteksi jatuh (Penting!)
      } else if (det.class_name.toLowerCase() === 'walking') {
        color = '#10b981'; // Hijau untuk berjalan
      }

      // Draw Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(rx, ry, rw, rh);

      // Draw Label Background
      ctx.fillStyle = color;
      const text = `${det.class_name.toUpperCase()} (${Math.round(det.confidence * 100)}%)`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(rx - 1.5, ry - 22, textWidth + 10, 22);

      // Draw Label Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(text, rx + 4, ry - 6);
    });
  };

  return (
    <div className="relative w-full overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/80 shadow-2xl">
      <canvas ref={canvasRef} className="block w-full h-auto" />
    </div>
  );
};

export default DetectionCanvas;
```

---

## 🏃 6. Cara Menjalankan Project secara Lokal

### Terminal 1: Run Backend (FastAPI)
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1   # Aktifkan virtualenv (Windows)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Terminal 2: Run Frontend (React Vite)
```bash
cd frontend
npm install
npm run dev
```
Buka browser Anda di `http://localhost:5173` untuk menguji aplikasi secara penuh.
