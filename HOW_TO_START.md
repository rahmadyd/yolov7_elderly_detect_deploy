# How To Start

Panduan ini menjelaskan cara menjalankan project YOLOv7 Elderly Detection Web App di local Windows menggunakan Anaconda environment `de_master`.

## 1. Pastikan Struktur Project

Pastikan kamu berada di root project:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy
```

Struktur penting project:

```txt
backend/api/main.py
backend/api/detector.py
backend/api/requirements.txt
backend/onnx/elderly.onnx
frontend/package.json
start-dev.ps1
```

## 2. Aktifkan Conda Environment

Project ini memakai conda environment:

```txt
de_master
```

Aktifkan env:

```powershell
conda activate de_master
```

## 3. Install Dependency Backend

Jalankan dari root project:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy\backend\api
pip install -r requirements.txt
```

Dependency backend dipakai untuk:

- FastAPI sebagai API server
- Uvicorn untuk menjalankan server
- ONNX Runtime untuk menjalankan model
- OpenCV dan NumPy untuk preprocessing gambar

## 4. Install Dependency Frontend

Masuk ke folder frontend:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy\frontend
npm install
```

## 5. Jalankan Backend dan Frontend Sekaligus

Kembali ke root project:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy
```

Jalankan:

```powershell
.\start-dev.ps1
```

Script ini akan membuka dua window PowerShell:

```txt
Backend  : http://localhost:8000
Frontend : http://localhost:5173
```

## 6. Buka Aplikasi

Buka browser:

```txt
http://localhost:5173
```

Langkah penggunaan:

1. Klik `Pilih Gambar`.
2. Pilih file gambar.
3. Klik `Detect`.
4. Hasil bounding box akan muncul di atas gambar.
5. History hasil deteksi akan muncul di tabel `Tracking CSV`.

## 7. Cek API Backend

Health check:

```txt
http://localhost:8000
```

Swagger UI:

```txt
http://localhost:8000/docs
```

Endpoint utama:

```txt
POST /detect/image
GET  /history
```

## 8. Lokasi File Tracking CSV

Hasil deteksi disimpan ke:

```txt
backend/api/history/detections.csv
```

Yang disimpan hanya metadata hasil deteksi:

- waktu deteksi
- nama file
- class
- confidence
- koordinat bounding box

Gambar yang diupload tidak disimpan.

## 9. Cara Stop Server

Tutup dua window PowerShell yang terbuka:

- window backend
- window frontend

## 10. Troubleshooting

### `npm` error karena PowerShell script disabled

Gunakan:

```powershell
npm.cmd run dev
```

Script `start-dev.ps1` sudah memakai `npm.cmd`, jadi biasanya aman.

### Backend tidak bisa import package

Pastikan env sudah aktif:

```powershell
conda activate de_master
```

Lalu install ulang:

```powershell
cd D:\projects\yolov7_elderly_detect_deploy\backend\api
pip install -r requirements.txt
```

### Model tidak ditemukan

Pastikan file model ada di:

```txt
backend/onnx/elderly.onnx
```

### Frontend tidak bisa detect

Pastikan backend menyala di:

```txt
http://localhost:8000
```

Lalu coba buka:

```txt
http://localhost:8000/docs
```

Kalau Swagger UI muncul, backend sudah hidup.
