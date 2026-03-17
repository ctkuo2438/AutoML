import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../api/client';

export default function PreprocessPage() {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  const [columns, setColumns] = useState<string[]>([]);
  const [dropColumns, setDropColumns] = useState<string[]>([]);
  type MissingStrategy = 'remove_column' | 'remove_rows' | 'fill_custom' | 'mean' | 'median' | 'mode' | 'drop';
  type ScalingMethod = 'standard' | 'minmax';

  const [fillMissing, setFillMissing] = useState<MissingStrategy>('mean');
  const [normalize, setNormalize] = useState(false);
  const [scalingMethod, setScalingMethod] = useState<ScalingMethod>('standard');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get(`/api/data/summary/${fileId}`)
      .then((res) => setColumns(res.data.summary.columns))
      .catch(() => setError('Failed to load columns'));
  }, [fileId]);

  const toggleDrop = (col: string) => {
    setDropColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const handleApply = async () => {
    setBusy(true);
    setError('');
    try {
      await api.post(`/api/data/preprocess/${fileId}`, {
        handle_missing: dropColumns.length > 0 || true,
        missing_config: { strategy: fillMissing, drop_columns: dropColumns },
        scale_features: normalize,
        scaling_config: normalize ? { method: scalingMethod } : undefined,
      });
      navigate(`/train/${fileId}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Preprocessing failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Preprocessing</h1>

        <div className="bg-white rounded-lg shadow p-6 mb-4">
          <h2 className="font-semibold text-gray-700 mb-3">Drop columns</h2>
          <div className="flex flex-wrap gap-2">
            {columns.map((col) => (
              <button
                key={col}
                onClick={() => toggleDrop(col)}
                className={`px-3 py-1 rounded-full text-sm border transition ${
                  dropColumns.includes(col)
                    ? 'bg-red-100 border-red-400 text-red-700'
                    : 'bg-gray-100 border-gray-300 text-gray-600 hover:border-gray-400'
                }`}
              >
                {col}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-4">
          <h2 className="font-semibold text-gray-700 mb-3">Handle missing values</h2>
          <select
            value={fillMissing}
            onChange={(e) => setFillMissing(e.target.value as MissingStrategy)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="mean">Fill with mean</option>
            <option value="median">Fill with median</option>
            <option value="mode">Fill with mode</option>
            <option value="drop">Drop rows with missing values</option>
            <option value="remove_column">Remove columns above missing threshold</option>
            <option value="remove_rows">Remove rows above missing threshold</option>
            <option value="fill_custom">Fill with custom value</option>
          </select>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="font-semibold text-gray-700 mb-3">Scale features</h2>
          <label className="flex items-center gap-2 cursor-pointer mb-3">
            <input
              type="checkbox"
              checked={normalize}
              onChange={(e) => setNormalize(e.target.checked)}
              className="w-4 h-4 accent-indigo-600"
            />
            <span className="text-sm text-gray-600">Apply feature scaling</span>
          </label>
          {normalize && (
            <select
              value={scalingMethod}
              onChange={(e) => setScalingMethod(e.target.value as ScalingMethod)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="standard">Standard (z-score)</option>
              <option value="minmax">Min-Max (0–1)</option>
            </select>
          )}
        </div>

        {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

        <div className="flex gap-3">
          <button
            onClick={() => navigate(`/preview/${fileId}`)}
            className="px-4 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
          >
            ← Back
          </button>
          <button
            onClick={handleApply}
            disabled={busy}
            className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition disabled:opacity-40"
          >
            {busy ? 'Applying...' : 'Apply & Train →'}
          </button>
        </div>
      </div>
    </div>
  );
}
