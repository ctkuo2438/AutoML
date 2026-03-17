import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';

export default function UploadPage() {
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const MAX_FILE_BYTES = 100 * 1024 * 1024; // 100 MB

  const handleFile = (f: File) => {
    if (!f.name.endsWith('.csv')) {
      setError('Only CSV files are supported.');
      return;
    }
    if (f.size > MAX_FILE_BYTES) {
      setError('File exceeds the 100 MB limit.');
      return;
    }
    setFile(f);
    setError('');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await api.post('/api/files/upload', formData);
      navigate(`/preview/${res.data.file_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-6">
      <div className="bg-white rounded-lg shadow-md w-full max-w-lg p-8">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Upload CSV</h1>
        <p className="text-gray-500 text-sm mb-6">
          Upload a CSV file to start the AutoML pipeline.
        </p>

        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition ${
            dragging ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-indigo-400'
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
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
            <>
              <p className="text-gray-400 text-sm">Drag & drop a CSV here</p>
              <p className="text-gray-400 text-xs mt-1">or click to browse</p>
            </>
          )}
        </div>

        {error && (
          <p className="text-red-600 text-sm mt-3">{error}</p>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="mt-6 w-full bg-indigo-600 text-white py-2 rounded-md font-medium hover:bg-indigo-700 transition disabled:opacity-40"
        >
          {uploading ? 'Uploading...' : 'Upload & Preview'}
        </button>

        <button
          onClick={() => navigate('/dashboard')}
          className="mt-3 w-full text-sm text-gray-500 hover:text-gray-700"
        >
          View training history →
        </button>
      </div>
    </div>
  );
}
