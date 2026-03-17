import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../api/client';

export default function TrainPage() {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  const [columns, setColumns] = useState<string[]>([]);
  const [experimentName, setExperimentName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get(`/api/data/summary/${fileId}`)
      .then((res) => {
        const cols = res.data.summary.columns as string[];
        setColumns(cols);
        if (cols.length > 0) setTargetColumn(cols[cols.length - 1]);
      })
      .catch(() => setError('Failed to load columns'));
  }, [fileId]);

  const handleTrain = async () => {
    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const res = await api.post('/api/training/train', {
        file_id: fileId,
        target_column: targetColumn,
        task_type: taskType,
        algorithm,
        experiment_name: experimentName.trim() || undefined,
      });
      navigate(`/metrics/${res.data.job_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Training failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Configure Training</h1>

        <div className="bg-white rounded-lg shadow p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Experiment name <span className="text-gray-400 font-normal">(optional)</span>
            </label>
            <input
              type="text"
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
              placeholder="e.g. Diabetes, Titanic"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Target column
            </label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Task type
            </label>
            <div className="flex gap-3">
              {(['classification', 'regression'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTaskType(t)}
                  className={`flex-1 py-2 rounded-md text-sm font-medium border transition ${
                    taskType === t
                      ? 'bg-indigo-600 text-white border-indigo-600'
                      : 'bg-white text-gray-600 border-gray-300 hover:border-indigo-400'
                  }`}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Algorithm
            </label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="random_forest">Random Forest</option>
              <option value="lightgbm">LightGBM</option>
              <option value="xgboost">XGBoost</option>
            </select>
          </div>
        </div>

        {error && <p className="text-red-600 text-sm mt-4">{error}</p>}

        <div className="flex gap-3 mt-6">
          <button
            onClick={() => navigate(`/preprocess/${fileId}`)}
            className="px-4 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
          >
            ← Back
          </button>
          <button
            onClick={handleTrain}
            disabled={busy}
            className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition disabled:opacity-40"
          >
            {busy ? 'Training...' : 'Start Training →'}
          </button>
        </div>
      </div>
    </div>
  );
}
