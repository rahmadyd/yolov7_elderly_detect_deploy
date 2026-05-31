import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  FileImage,
  Loader2,
  RefreshCw,
  Upload,
} from 'lucide-react'
import './index.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [detections, setDetections] = useState([])
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [error, setError] = useState('')

  const topDetection = useMemo(() => {
    if (!detections.length) return null
    return [...detections].sort((a, b) => b.confidence - a.confidence)[0]
  }, [detections])

  useEffect(() => {
    loadHistory()
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const loadHistory = async () => {
    setHistoryLoading(true)

    try {
      const response = await fetch(`${API_URL}/history`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Gagal mengambil history.')
      }

      setHistory(data.history || [])
    } catch (err) {
      setHistory([])
    } finally {
      setHistoryLoading(false)
    }
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (previewUrl) URL.revokeObjectURL(previewUrl)

    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setImageSize({ width: 0, height: 0 })
    setDetections([])
    setError('')
  }

  const detectImage = async () => {
    if (!selectedFile) {
      setError('Pilih gambar dulu sebelum deteksi.')
      return
    }

    setLoading(true)
    setError('')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch(`${API_URL}/detect/image`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Deteksi gagal.')
      }

      setDetections(data.detections || [])
      await loadHistory()
    } catch (err) {
      setError(err.message || 'Tidak bisa menghubungi backend.')
    } finally {
      setLoading(false)
    }
  }

  const resetDetection = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setSelectedFile(null)
    setPreviewUrl('')
    setImageSize({ width: 0, height: 0 })
    setDetections([])
    setError('')
  }

  const boxStyle = (box) => {
    const [x, y, width, height] = box
    return {
      left: `${(x / imageSize.width) * 100}%`,
      top: `${(y / imageSize.height) * 100}%`,
      width: `${(width / imageSize.width) * 100}%`,
      height: `${(height / imageSize.height) * 100}%`,
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-blue-600">
              YOLOv7 ONNX
            </p>
            <h1 className="mt-1 text-2xl font-bold sm:text-3xl">
              Elderly Detection Dashboard
            </h1>
          </div>
          <div className="flex items-center gap-2 rounded-full bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700">
            <Activity size={18} />
            API: {API_URL}
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 py-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(360px,0.9fr)]">
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-slate-300 px-4 py-3 text-sm font-semibold text-slate-700 transition hover:border-blue-400 hover:bg-blue-50">
                <Upload size={18} />
                Pilih Gambar
                <input
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  onChange={handleFileChange}
                />
              </label>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={detectImage}
                  disabled={loading || !selectedFile}
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {loading ? <Loader2 className="animate-spin" size={18} /> : <FileImage size={18} />}
                  Detect
                </button>
                <button
                  type="button"
                  onClick={resetDetection}
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                >
                  <RefreshCw size={18} />
                  Reset
                </button>
              </div>
            </div>

            {selectedFile && (
              <p className="mt-3 truncate text-sm text-slate-500">
                File: <span className="font-medium text-slate-700">{selectedFile.name}</span>
              </p>
            )}
          </div>

          {error && (
            <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              <AlertCircle className="mt-0.5 shrink-0" size={18} />
              <span>{error}</span>
            </div>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
            {previewUrl ? (
              <div className="relative overflow-hidden rounded-md bg-slate-950">
                <img
                  src={previewUrl}
                  alt="Preview deteksi"
                  className="block w-full"
                  onLoad={(event) =>
                    setImageSize({
                      width: event.currentTarget.naturalWidth,
                      height: event.currentTarget.naturalHeight,
                    })
                  }
                />

                {imageSize.width > 0 &&
                  detections.map((detection, index) => (
                    <div
                      key={`${detection.class_name}-${index}`}
                      className="absolute border-2 border-emerald-400 shadow-[0_0_0_1px_rgba(15,23,42,0.4)]"
                      style={boxStyle(detection.box)}
                    >
                      <span className="absolute left-0 top-0 -translate-y-full whitespace-nowrap rounded-t bg-emerald-500 px-2 py-1 text-xs font-bold text-white">
                        {detection.class_name} {(detection.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            ) : (
              <div className="flex min-h-[420px] flex-col items-center justify-center rounded-md border border-dashed border-slate-300 bg-slate-50 text-center">
                <FileImage className="text-slate-400" size={56} />
                <h2 className="mt-4 text-lg font-bold text-slate-800">Belum ada gambar</h2>
                <p className="mt-1 max-w-sm text-sm text-slate-500">
                  Upload gambar, klik Detect, lalu bounding box akan muncul di sini.
                </p>
              </div>
            )}
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-lg font-bold">Hasil Deteksi</h2>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Metric label="Total object" value={detections.length} />
              <Metric
                label="Top class"
                value={topDetection ? topDetection.class_name : '-'}
              />
            </div>

            <div className="mt-4 space-y-2">
              {detections.length ? (
                detections.map((detection, index) => (
                  <div
                    key={`${detection.class_name}-result-${index}`}
                    className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2 text-sm"
                  >
                    <span className="font-semibold capitalize text-slate-800">
                      {detection.class_name}
                    </span>
                    <span className="text-slate-500">
                      {(detection.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))
              ) : (
                <p className="rounded-md bg-slate-50 px-3 py-4 text-sm text-slate-500">
                  Hasil deteksi akan muncul setelah gambar diproses.
                </p>
              )}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-bold">Tracking CSV</h2>
              <button
                type="button"
                onClick={loadHistory}
                className="rounded-md border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100"
              >
                Refresh
              </button>
            </div>

            <div className="mt-4 max-h-[380px] overflow-auto rounded-md border border-slate-200">
              <table className="min-w-full text-left text-sm">
                <thead className="sticky top-0 bg-slate-100 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Waktu</th>
                    <th className="px-3 py-2">File</th>
                    <th className="px-3 py-2">Class</th>
                    <th className="px-3 py-2">Conf</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {historyLoading ? (
                    <tr>
                      <td colSpan="4" className="px-3 py-5 text-center text-slate-500">
                        Loading history...
                      </td>
                    </tr>
                  ) : history.length ? (
                    history
                      .slice()
                      .reverse()
                      .map((item, index) => (
                        <tr key={`${item.timestamp}-${item.filename}-${index}`}>
                          <td className="whitespace-nowrap px-3 py-2 text-slate-500">
                            {item.timestamp}
                          </td>
                          <td className="max-w-[120px] truncate px-3 py-2 font-medium">
                            {item.filename}
                          </td>
                          <td className="px-3 py-2 capitalize">{item.class_name}</td>
                          <td className="px-3 py-2">
                            {Number(item.confidence) > 0
                              ? `${(Number(item.confidence) * 100).toFixed(1)}%`
                              : '-'}
                          </td>
                        </tr>
                      ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="px-3 py-5 text-center text-slate-500">
                        Belum ada tracking.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </aside>
      </section>
    </main>
  )
}

function Metric({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4">
      <p className="text-xs font-semibold uppercase text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-slate-950">{value}</p>
    </div>
  )
}
