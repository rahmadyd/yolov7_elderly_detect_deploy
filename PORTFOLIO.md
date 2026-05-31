# YOLOv7 Elderly Detection Web App

## Ringkasan Proyek

YOLOv7 Elderly Detection Web App adalah aplikasi web untuk mendeteksi objek pada gambar menggunakan model custom YOLOv7 yang sudah dikonversi ke format ONNX. Aplikasi ini dibuat dengan arsitektur frontend dan backend terpisah: React sebagai antarmuka pengguna, FastAPI sebagai API server, dan ONNX Runtime sebagai engine inference model.

Proyek ini dirancang untuk mendeteksi class `lansia` dan `muda` dari gambar yang diunggah pengguna. Setelah gambar diproses, hasil deteksi ditampilkan langsung di web dalam bentuk bounding box, label class, dan confidence score. Gambar tidak disimpan di server, tetapi hasil deteksi dicatat ke file CSV sebagai tracking sederhana.

## Tujuan Proyek

Tujuan utama proyek ini adalah membangun pipeline deployment model computer vision dari model YOLOv7 ke aplikasi web yang dapat digunakan secara interaktif.

Fokus proyek:

- Menggunakan model YOLOv7 custom untuk deteksi gambar.
- Mengonversi model ke ONNX agar inference lebih ringan.
- Membuat API deteksi gambar menggunakan FastAPI.
- Membuat frontend React untuk upload gambar dan visualisasi bounding box.
- Menyimpan riwayat hasil deteksi ke CSV tanpa database.
- Membuat aplikasi yang sederhana, ringan, dan mudah dijalankan secara lokal.

## Teknologi yang Digunakan

### Backend

- Python
- FastAPI
- Uvicorn
- ONNX Runtime
- OpenCV
- NumPy
- CSV file sebagai tracking history

### Frontend

- React
- Vite
- Tailwind CSS
- Lucide React Icons
- Fetch API

### Model

- YOLOv7
- Custom trained model
- ONNX model format

## Struktur Project

```txt
yolov7_elderly_detect_deploy/
|
|-- backend/
|   |-- api/
|   |   |-- main.py
|   |   |-- detector.py
|   |   |-- history.py
|   |   |-- requirements.txt
|   |   `-- history/
|   |
|   |-- onnx/
|   |   |-- elderly.onnx
|   |   |-- test_onnx.py
|   |   `-- sample images / testing files
|   |
|   `-- yolov7/
|       `-- source YOLOv7 dan model training/export
|
|-- frontend/
|   |-- src/
|   |   |-- App.jsx
|   |   |-- main.jsx
|   |   `-- index.css
|   |
|   |-- package.json
|   |-- vite.config.js
|   `-- tailwind.config.js
|
|-- project_design_yolov7.md
|-- complete_project_blueprint.md
|-- .gitignore
`-- PORTFOLIO.md
```

## Alur Kerja Aplikasi

```txt
User upload gambar di React
        |
        v
Frontend mengirim gambar ke FastAPI
        |
        v
FastAPI menerima file gambar
        |
        v
Detector membaca gambar dari bytes
        |
        v
Gambar dipreprocess ke format input YOLOv7 ONNX
        |
        v
ONNX Runtime menjalankan inference
        |
        v
Backend melakukan postprocess dan NMS
        |
        v
Hasil deteksi dikembalikan sebagai JSON
        |
        v
Frontend menampilkan bounding box di atas gambar
        |
        v
Backend mencatat hasil deteksi ke CSV
```

## Fitur Utama

### 1. Upload Gambar

Pengguna dapat memilih gambar dari perangkat lokal melalui frontend React.

### 2. Deteksi Object dengan YOLOv7 ONNX

Backend menjalankan model `elderly.onnx` menggunakan ONNX Runtime. Model menerima gambar yang sudah diproses ke ukuran 640x640.

### 3. Bounding Box di Frontend

Hasil deteksi dikirim sebagai data JSON berupa class, confidence, dan koordinat bounding box. Frontend menggambar bounding box langsung di atas preview gambar.

### 4. Tracking CSV

Setiap hasil deteksi dicatat ke file CSV. Gambar tidak disimpan, hanya metadata hasil deteksi.

Data yang disimpan:

- timestamp
- nama file
- class terdeteksi
- confidence
- posisi bounding box

### 5. Halaman History

Frontend mengambil data dari endpoint `/history` dan menampilkannya sebagai tabel tracking.

## Endpoint API

### Health Check

```txt
GET /
```

Response:

```json
{
  "status": "ok",
  "message": "YOLOv7 Elderly Detection API is running"
}
```

### Deteksi Gambar

```txt
POST /detect/image
```

Request:

- multipart form-data
- field: `file`
- tipe file: image

Contoh response:

```json
{
  "status": "success",
  "filename": "sample.png",
  "detections": [
    {
      "class_id": 0,
      "class_name": "lansia",
      "confidence": 0.8742,
      "box": [120, 80, 240, 360]
    }
  ]
}
```

### History Deteksi

```txt
GET /history
```

Response:

```json
{
  "status": "success",
  "history": [
    {
      "timestamp": "2026-05-31 14:30:00",
      "filename": "sample.png",
      "class_name": "lansia",
      "confidence": "0.8742",
      "x": "120",
      "y": "80",
      "width": "240",
      "height": "360"
    }
  ]
}
```

## Cara Menjalankan Project

### 1. Jalankan Backend

Aktifkan environment Anaconda:

```powershell
conda activate nama_env_kamu
```

Masuk ke folder backend API:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy\backend\api
```

Install dependency jika belum:

```powershell
pip install -r requirements.txt
```

Jalankan FastAPI:

```powershell
uvicorn main:app --reload --port 8000
```

Backend berjalan di:

```txt
http://localhost:8000
```

### 2. Jalankan Frontend

Masuk ke folder frontend:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy\frontend
```

Install dependency jika belum:

```powershell
npm install
```

Jalankan Vite:

```powershell
npm.cmd run dev
```

Frontend berjalan di:

```txt
http://localhost:5173
```

## Cara Kerja Backend

Backend memiliki tiga file utama:

### `main.py`

Berisi konfigurasi FastAPI, CORS, endpoint upload gambar, dan endpoint history.

### `detector.py`

Berisi class `YOLOv7ONNXDetector` yang bertugas:

- load model ONNX
- decode image bytes
- resize gambar ke 640x640
- normalisasi gambar
- menjalankan inference
- melakukan postprocess
- menjalankan NMS
- mengembalikan hasil deteksi sebagai list JSON

### `history.py`

Berisi fungsi untuk menyimpan hasil deteksi ke CSV dan membaca kembali isi CSV untuk ditampilkan di frontend.

## Cara Kerja Frontend

Frontend berada di `frontend/src/App.jsx`.

Fungsi utama frontend:

- memilih gambar dari file input
- menampilkan preview gambar
- mengirim gambar ke backend
- menerima hasil deteksi
- menggambar bounding box menggunakan elemen HTML overlay
- menampilkan ringkasan hasil deteksi
- menampilkan tabel tracking dari CSV

## Kenapa Tidak Menggunakan Database?

Pada versi awal, aplikasi belum membutuhkan database karena data yang disimpan masih sederhana dan hanya digunakan sebagai tracking hasil testing.

Sebagai gantinya, aplikasi menggunakan CSV karena:

- lebih ringan
- mudah dibuat
- mudah dibuka di Excel
- cukup untuk MVP
- tidak membutuhkan setup database

Database seperti SQLite, MySQL, atau PostgreSQL bisa ditambahkan nanti jika aplikasi membutuhkan fitur seperti login, filter kompleks, penghapusan data, statistik dashboard, atau multi-user.

## Catatan Model

Model ONNX tidak disarankan dimasukkan ke Git karena ukurannya besar. File seperti `.onnx`, `.pt`, dan hasil deteksi sebaiknya masuk `.gitignore`.

File model utama:

```txt
backend/onnx/elderly.onnx
```

Script testing manual:

```txt
backend/onnx/test_onnx.py
```

Script tersebut digunakan untuk membuktikan bahwa model ONNX dapat melakukan inference. Setelah itu, logic pentingnya dipindahkan ke `backend/api/detector.py` agar bisa dipanggil oleh FastAPI.

## Nilai Portofolio

Proyek ini menunjukkan kemampuan dalam:

- membangun aplikasi full-stack sederhana
- deploy model computer vision ke web app
- membuat backend inference API
- menghubungkan React dengan FastAPI
- melakukan preprocessing dan postprocessing model YOLO
- menggunakan ONNX Runtime untuk inference
- membuat tracking hasil deteksi tanpa database
- mendesain workflow dari model machine learning ke aplikasi nyata

## Pengembangan Selanjutnya

Beberapa fitur yang bisa ditambahkan:

- deteksi realtime dari webcam
- upload video dan frame skipping
- dashboard statistik hasil deteksi
- export history ke Excel
- SQLite database untuk tracking lebih rapi
- autentikasi user
- deployment ke cloud
- optimasi model dengan quantization
- akselerasi GPU menggunakan ONNX Runtime DirectML

## Deskripsi Singkat untuk CV atau LinkedIn

Membangun aplikasi web deteksi lansia berbasis YOLOv7 dengan React, FastAPI, dan ONNX Runtime. Model custom YOLOv7 dikonversi ke ONNX untuk inference yang lebih ringan. Backend menerima upload gambar, menjalankan object detection, mengembalikan bounding box dalam format JSON, dan mencatat hasil deteksi ke CSV. Frontend menampilkan preview gambar, bounding box, confidence score, serta tabel history hasil deteksi.

## Deskripsi Singkat untuk GitHub

Full-stack YOLOv7 ONNX web application for elderly detection. Built with React, FastAPI, OpenCV, and ONNX Runtime. Supports image upload, object detection, bounding box visualization, and CSV-based detection history without storing uploaded images.
