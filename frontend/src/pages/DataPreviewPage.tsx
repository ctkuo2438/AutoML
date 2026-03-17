import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../api/client';

interface ColumnInfo {
  dtype: string;
  null_count: number;
  unique_count: number;
  is_numeric: boolean;
  mean?: number;
  min?: number;
  max?: number;
}

interface Summary {
  shape: { rows: number; columns: number };
  columns: string[];
  column_info: Record<string, ColumnInfo>;
  missing_values_total: number;
  duplicate_rows: number;
}

export default function DataPreviewPage() {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get(`/api/data/summary/${fileId}`)
      .then((res) => setSummary(res.data.summary))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load summary'));
  }, [fileId]);

  if (error) return <p className="p-8 text-red-600">{error}</p>;
  if (!summary) return <p className="p-8 text-gray-500">Loading...</p>;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-4">Data Preview</h1>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Rows</p>
            <p className="text-2xl font-bold text-indigo-600">{summary.shape.rows}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Columns</p>
            <p className="text-2xl font-bold text-indigo-600">{summary.shape.columns}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Missing Values</p>
            <p className="text-2xl font-bold text-yellow-500">{summary.missing_values_total}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Duplicate Rows</p>
            <p className="text-2xl font-bold text-red-500">{summary.duplicate_rows}</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow mb-6 overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-100 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-4 py-3">Column</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Missing</th>
                <th className="px-4 py-3">Unique</th>
                <th className="px-4 py-3">Min</th>
                <th className="px-4 py-3">Max</th>
                <th className="px-4 py-3">Mean</th>
              </tr>
            </thead>
            <tbody>
              {summary.columns.map((col) => {
                const info = summary.column_info[col];
                return (
                  <tr key={col} className="border-t border-gray-100">
                    <td className="px-4 py-2 font-medium">{col}</td>
                    <td className="px-4 py-2 text-gray-500">{info.dtype}</td>
                    <td className="px-4 py-2 text-gray-500">{info.null_count}</td>
                    <td className="px-4 py-2 text-gray-500">{info.unique_count}</td>
                    <td className="px-4 py-2 text-gray-500">
                      {info.min !== undefined ? info.min.toFixed(2) : '—'}
                    </td>
                    <td className="px-4 py-2 text-gray-500">
                      {info.max !== undefined ? info.max.toFixed(2) : '—'}
                    </td>
                    <td className="px-4 py-2 text-gray-500">
                      {info.mean !== undefined ? info.mean.toFixed(2) : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <button
          onClick={() => navigate(`/preprocess/${fileId}`)}
          className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition"
        >
          Next: Preprocess →
        </button>
      </div>
    </div>
  );
}
