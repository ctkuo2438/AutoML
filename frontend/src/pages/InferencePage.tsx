import { useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../api/client';

interface PredictionResponse {
  job_id: string;
  task_type: string;
  algorithm: string;
  num_rows: number;
  predictions: (number | string)[];
  data_with_predictions: Record<string, any>[];
  message: string;
}

export default function InferencePage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const handleFile = (f: File) => {
    if (!f.name.endsWith('.csv')) { setError('Only CSV files supported.'); return; }
    setFile(f);
    setError('');
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) return;
    setBusy(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const uploadRes = await api.post('/api/files/upload', formData);
      const inferFileId = uploadRes.data.file_id;

      const predRes = await api.post('/api/inference/predict', {
        job_id: jobId,
        file_id: inferFileId,
      });
      setResult(predRes.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Inference failed');
    } finally {
      setBusy(false);
    }
  };

  const downloadCSV = () => {
    if (!result) return;
    const rows = result.data_with_predictions;
    const cols = Object.keys(rows[0]);
    const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => r[c]).join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${jobId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Run Inference</h1>
        <p className="text-gray-500 text-sm mb-6">Upload a CSV with the same feature columns as your training data.</p>

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer mb-4 transition ${
            file ? 'border-indigo-400 bg-indigo-50' : 'border-gray-300 hover:border-indigo-400'
          }`}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          {file ? (
            <p className="text-indigo-700 font-medium">{file.name}</p>
          ) : (
            <p className="text-gray-400 text-sm">Click to select a CSV file</p>
          )}
        </div>

        {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

        <div className="flex gap-3 mb-8">
          <button
            onClick={() => navigate('/dashboard')}
            className="px-4 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
          >
            ← Dashboard
          </button>
          <button
            onClick={handlePredict}
            disabled={!file || busy}
            className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition disabled:opacity-40"
          >
            {busy ? 'Predicting...' : 'Predict'}
          </button>
        </div>

        {result && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-semibold text-gray-700">
                  {result.algorithm} · {result.task_type} · {result.num_rows} rows
                </p>
                <p className="text-sm text-green-600">{result.message}</p>
              </div>
              <button
                onClick={downloadCSV}
                className="text-sm text-indigo-600 border border-indigo-300 px-3 py-1 rounded-md hover:bg-indigo-50"
              >
                Download CSV
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-gray-50 text-gray-500 uppercase">
                  <tr>
                    {Object.keys(result.data_with_predictions[0]).map((c) => (
                      <th key={c} className={`px-3 py-2 ${c === 'prediction' ? 'text-indigo-600' : ''}`}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.data_with_predictions.slice(0, 20).map((row, i) => (
                    <tr key={i} className="border-t border-gray-100">
                      {Object.entries(row).map(([k, v]) => (
                        <td
                          key={k}
                          className={`px-3 py-2 ${k === 'prediction' ? 'font-semibold text-indigo-700' : ''}`}
                        >
                          {String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.num_rows > 20 && (
                <p className="text-xs text-gray-400 px-3 py-2">
                  Showing 20 of {result.num_rows} rows. Download CSV for full results.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
