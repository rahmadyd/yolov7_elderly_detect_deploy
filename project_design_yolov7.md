# Panduan Arsitektur & Implementasi: YOLOv7 Elderly Detection Web App
Dokumen ini berisi rancangan arsitektur, alur kerja (workflow), struktur folder, kode implementasi, serta strategi optimasi untuk mendeploy model custom YOLOv7 ke dalam aplikasi web berbasis **React (Frontend)** dan **FastAPI + ONNX Runtime (Backend)** yang ringan dan cocok untuk laptop spesifikasi rendah (low-resource).

---

## 1. Arsitektur Project (Clean Architecture)

Untuk menjaga kode tetap modular, testable, dan mudah dirawat, kita memisahkan tanggung jawab (Separation of Concerns) menggunakan konsep **Clean Architecture**.

```mermaid
graph TD
    subgraph Frontend (React + Vite)
        UI[React Components / UI] --> Canvas[Canvas Draw Engine]
        UI --> API_Client[API Fetch/Axios Client]
    end

    subgraph Backend (FastAPI)
        Router[API Router / Endpoints] --> Controller[Inference Controller]
        Controller --> UseCase[Object Detection Use Case]
        UseCase --> Service[ONNX Inference Service]
        Service --> PrePost[Image Pre/Post-processing]
        Service --> Model[(YOLOv7 ONNX Model)]
    end

    API_Client -- HTTP POST (Image/Video) --> Router
    Router -- JSON (Bounding Boxes Coordinates & Class) --> API_Client
```

### Kunci Ringan untuk Laptop Low-Resource:
*   **Decoupled Bounding Box Drawing**: Backend **TIDAK** menggambar bounding box pada gambar/video menggunakan OpenCV lalu mengirimkan kembali gambar yang berat ke Frontend. 
*   Backend hanya memproses model AI dan mengirimkan data **JSON koordinat (x, y, w, h, confidence, class)**.
*   Frontend (React) bertugas merender gambar asli dan menggambar bounding box secara real-time di atas elemen HTML `<canvas>`. Ini menghemat CPU backend dan bandwidth jaringan secara signifikan.

---

## 2. Struktur Folder Final (Modular Structure)

Berikut adalah struktur folder modular yang clean untuk memudahkan transisi dari development ke production:

```txt
yolov7-elderly-detect/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # Entry point FastAPI & CORS setup
│   │   │
│   │   ├── api/                    # Delivery Layer (HTTP/Web)
│   │   │   ├── __init__.py
│   │   │   ├── routes.py           # Endpoints untuk image & video upload
│   │   │
│   │   ├── core/                   # Configuration & Constants
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │
│   │   ├── services/               # Domain & Logic Layer (Inference)
│   │   │   ├── __init__.py
│   │   │   ├── detector.py         # Kelas utama ONNX Runtime wrapper
│   │   │   ├── utils.py            # Preprocessing & Postprocessing (NMS, Resize)
│   │   │
│   │   └── weights/                # Penyimpanan Model AI (.onnx)
│   │       └── yolov7-elderly.onnx
│   │
│   ├── requirements.txt            # Dependency Backend
│   └── Dockerfile                  # (Optional) Production Dockerfile
│
└── frontend/
    ├── src/
    │   ├── assets/                 # Logo, Global CSS
    │   ├── components/             # Reusable UI Components
    │   │   ├── Header.jsx
    │   │   ├── DetectionCanvas.jsx # Canvas untuk merender gambar + bounding boxes
    │   │   └── UploadArea.jsx      # Area dropzone untuk file
    │   │
    │   ├── hooks/                  # Custom React Hooks
    │   │   └── useDetection.js     # State management upload & API call
    │   │
    │   ├── App.jsx                 # Layout & Main Page
    │   ├── main.jsx                # Entry point React
    │   └── index.css               # Styling (Tailwind/Custom CSS)
    │
    ├── package.json                # Dependency Frontend
    ├── vite.config.js              # Konfigurasi Vite
    └── tailwind.config.js          # Konfigurasi Tailwind CSS
```

---

## 3. Workflow Deployment YOLOv7

Alur dari pembuatan model hingga running di production:

```
[PyTorch: best.pt] ➔ [Convert: export.py] ➔ [ONNX Model] ➔ [FastAPI Inference] ➔ [React UI]
```

1.  **Training**: Selesaikan training YOLOv7 di Google Colab atau Local PyTorch, menghasilkan file `best.pt`.
2.  **Export**: Lakukan konversi PyTorch model (`.pt`) ke ONNX format (`.onnx`).
3.  **Inference Engine**: Jalankan file `.onnx` menggunakan **ONNX Runtime (CPU/GPU)** di FastAPI.
4.  **UI Integration**: Tampilkan antarmuka upload file di React, kirim ke API backend, lalu gambar hasil deteksi di Canvas React.

---

## 4. Convert `best.pt` ke ONNX

ONNX (Open Neural Network Exchange) membuat model berjalan lebih cepat dan portable tanpa perlu library PyTorch yang sangat berat (PyTorch berukuran > 2 GB, sedangkan ONNX Runtime hanya ~50-80 MB).

### Langkah Konversi:
Gunakan repositori resmi YOLOv7 untuk mengekspor model. Jalankan perintah ini di environment PyTorch tempat Anda melatih model:

```bash
python export.py --weights weights/best.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.25 --img-size 640 640
```

### Penjelasan Parameter:
*   `--grid`: Mengekspor deteksi grid langsung ke dalam grafik ONNX (sangat mempermudah postprocessing).
*   `--end2end`: Menambahkan modul NMS (Non-Maximum Suppression) langsung ke dalam graf model (OPS / Operator). Ini membuat postprocessing di backend menjadi sangat efisien dan mudah dikodekan.
*   `--simplify`: Mengoptimalkan graf komputasi ONNX dengan menghapus layer yang redundan (memakai library `onnx-simplifier`).
*   `--img-size 640 640`: Menentukan input tensor size default.

Setelah selesai, Anda akan mendapatkan file `best.onnx` (atau `yolov7-elderly.onnx`) berukuran sekitar ~30-40 MB.

---

## 5. Setup ONNX Runtime

ONNX Runtime adalah engine inference performa tinggi. 

### Instalasi Dependency Backend (`requirements.txt`):
```text
fastapi==0.110.0
uvicorn==0.28.0
python-multipart==0.0.9
onnxruntime==1.17.1
opencv-python-headless==4.9.0.80
numpy==1.26.4
pillow==10.2.0
```
> **Catatan**: Menggunakan `opencv-python-headless` lebih disukai untuk deployment karena tidak memerlukan library GUI sistem operasi (seperti GTK/QT) yang sering menyebabkan error di server atau Docker.

---

## 6. Setup FastAPI Backend Inference

Berikut adalah implementasi clean code untuk kelas pendeteksi (`backend/app/services/detector.py`):

```python
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

class YOLOv7Detector:
    def __init__(self, model_path: str):
        # Konfigurasi session untuk menghemat resource CPU
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = 2  # Batasi thread agar tidak membebani CPU laptop
        self.opts.inter_op_num_threads = 2
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load model ONNX ke memory
        self.session = ort.InferenceSession(model_path, self.opts, providers=['CPUExecutionProvider'])
        
        # Ambil nama input dan output layer
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Detail input size
        self.input_shape = self.session.get_inputs()[0].shape # [batch, channels, height, width]
        self.input_height = self.input_shape[2] # 640
        self.input_width = self.input_shape[3]  # 640
        
        # Daftar kelas (sesuaikan dengan dataset custom Anda)
        self.classes = ["elderly", "fall", "walking"] # Contoh kelas deteksi lansia

    def preprocess(self, image: Image.Image):
        # 1. Konversi PIL Image ke Numpy Array
        img_np = np.array(image.convert("RGB"))
        
        # Simpan dimensi asli untuk kalkulasi bounding box nanti
        original_height, original_width = img_np.shape[:2]
        
        # 2. Resize gambar ke input size model (640x640)
        img_resized = cv2.resize(img_np, (self.input_width, self.input_height))
        
        # 3. Transpose format dari HWC (Height, Width, Channel) ke CHW (Channel, Height, Width)
        img_transposed = img_resized.transpose(2, 0, 1)
        
        # 4. Normalisasi nilai pixel ke range [0.0, 1.0]
        img_normalized = img_transposed.astype(np.float32) / 255.0
        
        # 5. Tambahkan dimensi batch [1, C, H, W]
        input_tensor = np.expand_dims(img_normalized, axis=0)
        
        return input_tensor, original_width, original_height

    def detect(self, image_bytes: bytes):
        # Load byte stream ke PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocessing
        input_tensor, orig_w, orig_h = self.preprocess(image)
        
        # Jalankan Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocessing
        detections = self.postprocess(outputs, orig_w, orig_h)
        
        return detections

    def postprocess(self, outputs, orig_w, orig_h):
        predictions = outputs[0]  # Shape biasanya [1, num_predictions, 7] atau format output export end2end
        
        detections = []
        
        # Struktur output YOLOv7 End2End ONNX biasanya: [batch_id, x1, y1, x2, y2, class_id, score]
        # Namun jika output formatnya default, deteksi dilewatkan secara terstruktur.
        for pred in predictions[0]:
            # Nilai confidence / score
            score = float(pred[6]) if len(pred) > 6 else float(pred[4]) # sesuaikan indeks output model anda
            
            if score < 0.25:  # Confidence threshold
                continue
                
            class_id = int(pred[5])
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            
            # Koordinat bounding box (ternormalisasi ke 640x640)
            # Skalakan kembali ke resolusi asli gambar
            x1 = float(pred[1]) / self.input_width * orig_w
            y1 = float(pred[2]) / self.input_height * orig_h
            x2 = float(pred[3]) / self.input_width * orig_w
            y2 = float(pred[4]) / self.input_height * orig_h
            
            # Format width dan height
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

---

## 7. Upload Image/Video Endpoint

Berikut adalah setup server FastAPI utama (`backend/app/main.py`):

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.detector import YOLOv7Detector
import os

app = FastAPI(title="YOLOv7 Elderly Detection API", version="1.0.0")

# Izinkan CORS agar React Frontend bisa berkomunikasi dengan FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti dengan URL frontend di production (misal: http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi detector
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "yolov7-elderly.onnx")
detector = None

@app.on_event("startup")
def load_model():
    global detector
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model file tidak ditemukan di {WEIGHTS_PATH}. Harap letakkan file ONNX Anda di sana.")
    detector = YOLOv7Detector(WEIGHTS_PATH)

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
        
    try:
        contents = await file.read()
        detections = detector.detect(contents)
        return {"status": "success", "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

@app.post("/detect/video-frame")
async def detect_frame(file: UploadFile = File(...)):
    # Berguna jika frontend mengirimkan frame video secara real-time (misalnya dari webcam)
    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="Frame harus berupa gambar.")
    try:
        contents = await file.read()
        detections = detector.detect(contents)
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 8. Preprocessing & Postprocessing Detection

Detail langkah yang terjadi pada backend secara internal:
1.  **Resize**: Gambar asli di-resize ke `640x640`.
2.  **Color Space**: OpenCV membaca gambar dalam format BGR secara default, namun PIL membacanya dalam format RGB. Kita pastikan format input adalah **RGB** agar akurasi model tidak turun.
3.  **Transpose**: Format data diubah dari `[H, W, C]` (misal: 640x640x3) ke `[C, H, W]` (3x640x640).
4.  **Scale**: Membagi semua nilai piksel dengan `255.0` agar bernilai `0.0 - 1.0`.
5.  **Scaling back coordinates**:
    $$\text{Scale Factor Width} = \frac{\text{Lebar Asli}}{\text{Lebar Input (640)}}$$
    $$\text{Scale Factor Height} = \frac{\text{Tinggi Asli}}{\text{Tinggi Input (640)}}$$
    Koordinat bounding box hasil prediksi dikalikan dengan faktor skala ini agar pas saat digambar di atas gambar resolusi asli.

---

## 9. Setup React Frontend

Gunakan Vite untuk menginisialisasi frontend React dengan cepat.

```bash
# Di root folder project
npx -y create-vite@latest frontend --template react
cd frontend
npm install
npm install lucide-react # library icon yang clean & modern
```

### Install Tailwind CSS untuk UI yang Premium:
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```
Sesuaikan isi file `tailwind.config.js` untuk memindai file React:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```
Tambahkan direktif Tailwind ke `frontend/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  background-color: #0f172a; /* Dark sleek mode */
  color: #f8fafc;
  font-family: 'Inter', sans-serif;
}
```

---

## 10. Upload Image dari React ke FastAPI

Gunakan React hooks modern untuk mengelola pengiriman data ke server API (`frontend/src/hooks/useDetection.js`):

```javascript
import { useState } from 'react';

export const useDetection = () => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const uploadAndDetect = async (file) => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setImage(file);
    setPreviewUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/detect/image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Gagal memproses gambar dari server.');
      }

      const data = await response.json();
      if (data.status === 'success') {
        setDetections(data.detections);
      } else {
        throw new Error('Format output deteksi tidak valid.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreviewUrl(null);
    setDetections([]);
    setError(null);
  };

  return { previewUrl, detections, loading, error, uploadAndDetect, reset };
};
```

---

## 11. Menampilkan Hasil Detection di Frontend (Canvas Rendering)

Ini adalah component krusial (`frontend/src/components/DetectionCanvas.jsx`) yang bertugas menggambar bounding box di atas gambar asli secara dinamis:

```jsx
import React, { useRef, useEffect } from 'react';

const DetectionCanvas = ({ previewUrl, detections }) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    if (!previewUrl) return;
    
    const image = new Image();
    image.src = previewUrl;
    imageRef.current = image;
    
    image.onload = () => {
      drawCanvas();
    };
  }, [previewUrl, detections]);

  // Handle responsivitas layar saat di-resize
  useEffect(() => {
    window.addEventListener('resize', drawCanvas);
    return () => window.removeEventListener('resize', drawCanvas);
  }, [detections]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    
    // Tentukan lebar render di layar (responsive container)
    const displayWidth = canvas.parentElement.clientWidth;
    const scale = displayWidth / img.naturalWidth;
    const displayHeight = img.naturalHeight * scale;

    // Atur dimensi canvas internal agar sama dengan tampilan render css
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Bersihkan canvas dan gambar ulang gambar aslinya
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

    // Gambar Bounding Box
    detections.forEach((det) => {
      const [x, y, w, h] = det.box;
      
      // Skalakan koordinat box dari koordinat gambar asli ke koordinat tampilan canvas
      const renderX = x * scale;
      const renderY = y * scale;
      const renderW = w * scale;
      const renderH = h * scale;

      // Beri warna berbeda tiap kelas agar menarik
      let boxColor = '#10b981'; // Hijau default
      if (det.class_name.toLowerCase() === 'fall') {
        boxColor = '#ef4444'; // Merah jika mendeteksi jatuh
      } else if (det.class_name.toLowerCase() === 'elderly') {
        boxColor = '#3b82f6'; // Biru untuk lansia biasa
      }

      // Draw Box
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(renderX, renderY, renderW, renderH);

      // Draw Label Background
      ctx.fillStyle = boxColor;
      const labelText = `${det.class_name} (${Math.round(det.confidence * 100)}%)`;
      ctx.font = 'bold 14px Inter, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      
      // Menggambar background label di atas box
      ctx.fillRect(renderX - 1.5, renderY - 25, textWidth + 12, 25);

      // Draw Label Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, renderX + 4, renderY - 8);
    });
  };

  return (
    <div className="relative w-full overflow-hidden rounded-xl border border-slate-700 bg-slate-900/50 shadow-2xl">
      <canvas ref={canvasRef} className="block w-full h-auto" />
    </div>
  );
};

export default DetectionCanvas;
```

---

## 12. Optimasi Inference untuk Laptop Low-Resource

Agar CPU laptop tidak mengalami spike atau overload 100% saat inference, ikuti beberapa best practice berikut:

1.  **Gunakan ONNX CPU Thread Limiting**:
    Secara default, ONNX Runtime menggunakan seluruh core CPU yang tersedia. Batasi penggunaannya di FastAPI dengan parameter berikut:
    ```python
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 2
    ```
2.  **ONNX Quantization**:
    Mengubah bobot data dari format Float32 (32-bit) ke INT8 (8-bit). Ini memangkas ukuran model hingga 4x lebih kecil (misal ~35MB menjadi ~8MB) dan mempercepat pemrosesan CPU hingga 2-3 kali lipat dengan penurunan akurasi yang sangat minimal (<1%).
    
    *Script untuk quantize model ONNX:*
    ```python
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    quantize_dynamic(
        model_input="yolov7-elderly.onnx",
        model_output="yolov7-elderly-quant.onnx",
        weight_type=QuantType.QUInt8
    )
    ```
3.  **Proses Frame Skip untuk Input Video**:
    Jika memproses file video atau streaming webcam, jangan lakukan inference di setiap frame (30fps). Gunakan teknik **Frame Skipping**: kirim frame ke backend hanya setiap 3 atau 5 frame sekali (misalnya 6fps). Pergerakan manusia masih sangat mudah dideteksi pada 6fps, sementara beban CPU berkurang hingga 80%.

---

## 13. Best Practice Deployment

*   **Penyimpanan Model**: Jangan push model berukuran besar ke Git. Gunakan Git LFS, atau download model menggunakan script bash saat build time.
*   **Production Server**: Gunakan `gunicorn` dengan Uvicorn worker untuk backend FastAPI jika dirilis di platform VPS Linux:
    ```bash
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
    ```
*   **Static File Serving**: Build React frontend menjadi static file (`npm run build`), lalu serve menggunakan Nginx. Nginx juga bertugas mengarahkan (reverse proxy) endpoint `/api/*` ke backend FastAPI.

---

## 14. Common Error & Cara Debugging

| Error | Penyebab Utama | Cara Mengatasi |
| :--- | :--- | :--- |
| `IndexError: index out of range` di Postprocessing | Output shape model hasil export ONNX berbeda dengan struktur parsing kode. | Cetak shape output (`print(outputs[0].shape)`) untuk melihat strukturnya, pastikan apakah NMS bawaan ikut ter-export atau tidak. |
| `CORS Error` pada Browser | React Frontend memanggil endpoint FastAPI tanpa izin Origin header. | Pastikan `CORSMiddleware` di FastAPI sudah dikonfigurasi dengan mengizinkan domain asal React (misal: `http://localhost:5173`). |
| Laptop hang / CPU 100% | Model terlalu kompleks atau thread ONNX Runtime terlalu banyak. | Pastikan model di-export ke resolusi 640x640 (bukan 1280), terapkan optimasi Thread Limiter, dan lakukan Quantization. |

---

## 15. Cara Testing Project

### Uji Coba Backend (FastAPI Swagger UI):
1.  Buka browser dan akses ke [http://localhost:8000/docs](http://localhost:8000/docs).
2.  Cari endpoint `/detect/image`.
3.  Klik tombol **Try it out**, upload gambar contoh, lalu klik **Execute**.
4.  Pastikan respon JSON mengembalikan list bounding box dengan status code `200 OK`.

### Uji Coba Unit Test (Python):
Buat test file sederhana untuk memverifikasi load model (`backend/tests/test_detector.py`):
```python
import os
from app.services.detector import YOLOv7Detector

def test_load_detector():
    weights = "app/weights/yolov7-elderly.onnx"
    if os.path.exists(weights):
        detector = YOLOv7Detector(weights)
        assert detector.session is not None
        print("Test Load Model: PASSED")
```

---

## 16. Cara Menjalankan Project (Lokal)

### Langkah 1: Jalankan Backend
1.  Buka terminal baru di root folder project.
2.  Masuk ke direktori backend:
    ```bash
    cd backend
    ```
3.  Buat virtual environment dan aktifkan:
    ```bash
    python -m venv venv
    # Di Windows Powershell:
    .\venv\Scripts\Activate.ps1
    # Di Linux/macOS:
    source venv/bin/activate
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Jalankan server development:
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

### Langkah 2: Jalankan Frontend
1.  Buka terminal baru yang lain.
2.  Masuk ke direktori frontend:
    ```bash
    cd frontend
    ```
3.  Install dependencies Node.js:
    ```bash
    npm install
    ```
4.  Jalankan server development:
    ```bash
    npm run dev
    ```
5.  Buka browser Anda dan arahkan ke alamat yang diberikan oleh Vite (biasanya `http://localhost:5173`).

---

## 17. Cara Deploy Project Nanti (Cloud)

Ketika Anda siap untuk merilis aplikasi ini ke publik (Cloud Deployment), berikut opsi paling efisien untuk budget terbatas:

1.  **Backend Deployment (FastAPI)**:
    *   **Render.com** / **Railway.app**: Sangat ramah pemula dan menyediakan free/cheap tier. Gunakan Dockerfile untuk membungkus kode backend Anda.
    *   **Hugging Face Spaces (Docker Space)**: Menyediakan komputasi gratis yang cukup stabil untuk model ONNX skala kecil.
2.  **Frontend Deployment (React Static)**:
    *   **Vercel** / **Netlify** / **GitHub Pages**: Gratis, cepat, dan terintegrasi otomatis dengan repository GitHub Anda.
3.  **Dockerisasi (Docker Compose)**:
    Untuk deployment satu tombol, Anda dapat membuat file `docker-compose.yml` untuk mengorkestrasi container Frontend Nginx dan Backend FastAPI secara bersamaan.

---

## Urutan Pengerjaan yang Paling Efektif
Agar pengerjaan tidak membingungkan, selesaikan dengan urutan ini:
1.  **Ekspor model** `.pt` Anda ke `.onnx` dan coba quantize model tersebut terlebih dahulu.
2.  Buat folder **backend** FastAPI, buat virtualenv, install dependencies, lalu buat script `detector.py` dan lakukan pengujian dengan Python script biasa untuk memastikan bounding box keluar dengan benar.
3.  Buat API Endpoint di FastAPI untuk menerima file upload gambar. Test menggunakan **Swagger UI (/docs)**.
4.  Inisialisasi project **React** menggunakan Vite.
5.  Buat component UI Upload area & Canvas rendering di React.
6.  Hubungkan frontend dengan backend menggunakan `fetch` API.
7.  Uji coba integrasi secara penuh menggunakan gambar lansia.
8.  Tambahkan optimasi opsional seperti input frame-skip untuk webcam / video.
