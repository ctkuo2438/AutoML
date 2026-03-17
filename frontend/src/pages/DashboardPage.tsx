import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import { useAuth } from '../context/AuthContext';

interface TrainingResponse {
  job_id: string;
  file_id: string;
  target_column?: string;
  task_type: string;
  algorithm: string;
  status: string;
  metrics: Record<string, any> | null;
  message: string;
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [jobs, setJobs] = useState<TrainingResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/training/jobs?page=1&page_size=50')
      .then((res) => setJobs(res.data.jobs ?? []))
      .catch(() => setJobs([]))
      .finally(() => setLoading(false));
  }, []);

  const statusColor = (s: string) => {
    if (s === 'completed') return 'text-green-600 bg-green-50';
    if (s === 'failed') return 'text-red-600 bg-red-50';
    return 'text-yellow-600 bg-yellow-50';
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-800">Training History</h1>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/upload')}
              className="bg-indigo-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-indigo-700 transition"
            >
              + New Training
            </button>
            <button
              onClick={() => { logout(); navigate('/'); }}
              className="text-sm text-gray-500 hover:text-gray-700 px-3 py-2"
            >
              Logout
            </button>
          </div>
        </div>

        {loading ? (
          <p className="text-gray-500">Loading...</p>
        ) : jobs.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-8 text-center text-gray-500">
            No training jobs yet.{' '}
            <button
              className="text-indigo-600 hover:underline"
              onClick={() => navigate('/upload')}
            >
              Upload a CSV to get started.
            </button>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-100 text-gray-600 uppercase text-xs">
                <tr>
                  <th className="px-4 py-3">Algorithm</th>
                  <th className="px-4 py-3">Task</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Score</th>
                  <th className="px-4 py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => {
                  const m = job.metrics ?? {};
                  const score = job.task_type === 'classification'
                    ? (m.accuracy != null ? `Acc: ${m.accuracy.toFixed(3)}` : '—')
                    : (m.rmse != null ? `RMSE: ${m.rmse.toFixed(3)}` : '—');
                  return (
                    <tr key={job.job_id} className="border-t border-gray-100 hover:bg-gray-50">
                      <td className="px-4 py-3 font-medium capitalize">
                        {job.algorithm.replace('_', ' ')}
                      </td>
                      <td className="px-4 py-3 capitalize">{job.task_type}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${statusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-500">{score}</td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <button
                            onClick={() => navigate(`/metrics/${job.job_id}`)}
                            className="text-indigo-600 hover:underline text-xs"
                          >
                            Metrics
                          </button>
                          {job.status === 'completed' && (
                            <button
                              onClick={() => navigate(`/inference/${job.job_id}`)}
                              className="text-green-600 hover:underline text-xs"
                            >
                              Infer
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
