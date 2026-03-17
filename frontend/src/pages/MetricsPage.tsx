import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../api/client';

interface TrainingResponse {
  job_id: string;
  file_id: string;
  task_type: string;
  algorithm: string;
  status: string;
  metrics: Record<string, any> | null;
  message: string;
}

function MetricCard({ label, value }: { label: string; value: number | undefined }) {
  if (value === undefined || value === null) return null;
  return (
    <div className="bg-white rounded-lg shadow p-4 text-center">
      <p className="text-sm text-gray-500 mb-1">{label}</p>
      <p className="text-2xl font-bold text-indigo-600">{value.toFixed(4)}</p>
    </div>
  );
}

export default function MetricsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<TrainingResponse | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get(`/api/training/jobs/${jobId}`)
      .then((res) => setJob(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load metrics'));
  }, [jobId]);

  if (error) return <p className="p-8 text-red-600">{error}</p>;
  if (!job) return <p className="p-8 text-gray-500">Loading...</p>;

  const m = job.metrics ?? {};
  const isClassification = job.task_type === 'classification';

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-800">Training Results</h1>
          <p className="text-gray-500 text-sm mt-1">
            {job.algorithm} · {job.task_type} ·{' '}
            <span className={`font-medium ${job.status === 'completed' ? 'text-green-600' : 'text-yellow-600'}`}>
              {job.status}
            </span>
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {isClassification ? (
            <>
              <MetricCard label="Accuracy" value={m.accuracy} />
              <MetricCard label="F1 Score" value={m.f1_score} />
              <MetricCard label="Precision" value={m.precision} />
              <MetricCard label="Recall" value={m.recall} />
            </>
          ) : (
            <>
              <MetricCard label="RMSE" value={m.rmse} />
              <MetricCard label="MAE" value={m.mae} />
              <MetricCard label="R²" value={m.r2_score} />
            </>
          )}
        </div>

        {isClassification && m.confusion_matrix && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="font-semibold text-gray-700 mb-4">Confusion Matrix</h2>
            <div className="overflow-x-auto">
              {/* For binary (2x2) matrices show TP/FP/FN/TN labels; otherwise show Predicted/Actual headers */}
              {(m.confusion_matrix as number[][]).length === 2 ? (
                <table className="text-sm border-collapse">
                  <thead>
                    <tr>
                      <td className="px-3 py-2" />
                      <td className="px-3 py-2" />
                      <td colSpan={2} className="px-3 py-2 text-center text-xs font-semibold text-gray-500 uppercase tracking-wide">
                        Predicted
                      </td>
                    </tr>
                    <tr>
                      <td className="px-3 py-2" />
                      <td className="px-3 py-2" />
                      <td className="border border-gray-200 px-4 py-2 text-center text-xs font-semibold text-gray-500">Negative (0)</td>
                      <td className="border border-gray-200 px-4 py-2 text-center text-xs font-semibold text-gray-500">Positive (1)</td>
                    </tr>
                  </thead>
                  <tbody>
                    {(['Negative (0)', 'Positive (1)'] as const).map((rowLabel, i) => (
                      <tr key={i}>
                        {i === 0 && (
                          <td
                            rowSpan={2}
                            className="px-3 py-2 text-center text-xs font-semibold text-gray-500 uppercase tracking-wide"
                            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                          >
                            Actual
                          </td>
                        )}
                        <td className="border border-gray-200 px-4 py-2 text-center text-xs font-semibold text-gray-500">
                          {rowLabel}
                        </td>
                        {(m.confusion_matrix as number[][])[i].map((v, j) => {
                          const label = i === 0 && j === 0 ? 'TN' : i === 0 && j === 1 ? 'FP' : i === 1 && j === 0 ? 'FN' : 'TP';
                          const isCorrect = i === j;
                          return (
                            <td
                              key={j}
                              className={`border border-gray-200 px-4 py-3 text-center font-mono ${
                                isCorrect ? 'bg-indigo-50 text-indigo-700 font-bold' : 'bg-red-50 text-red-600 font-bold'
                              }`}
                            >
                              <div className="text-lg">{v}</div>
                              <div className="text-xs font-semibold mt-0.5 opacity-70">{label}</div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <table className="text-sm border-collapse">
                  <thead>
                    <tr>
                      <td className="px-3 py-2" />
                      <td colSpan={(m.confusion_matrix as number[][])[0].length} className="px-3 py-2 text-center text-xs font-semibold text-gray-500 uppercase tracking-wide">
                        Predicted
                      </td>
                    </tr>
                  </thead>
                  <tbody>
                    {(m.confusion_matrix as number[][]).map((row, i) => (
                      <tr key={i}>
                        {i === 0 && (
                          <td
                            rowSpan={(m.confusion_matrix as number[][]).length}
                            className="px-3 py-2 text-center text-xs font-semibold text-gray-500 uppercase tracking-wide"
                            style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                          >
                            Actual
                          </td>
                        )}
                        {row.map((v, j) => (
                          <td
                            key={j}
                            className={`border border-gray-200 px-4 py-2 text-center font-mono ${
                              i === j ? 'bg-indigo-50 text-indigo-700 font-bold' : 'text-gray-600'
                            }`}
                          >
                            {v}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={() => navigate('/dashboard')}
            className="px-4 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
          >
            View History
          </button>
          <button
            onClick={() => navigate(`/inference/${job.job_id}`)}
            className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition"
          >
            Run Inference →
          </button>
        </div>
      </div>
    </div>
  );
}
