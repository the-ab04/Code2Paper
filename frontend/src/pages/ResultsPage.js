import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getRun } from '../api/client';
import PaperPreview from '../components/PaperPreview';

export default function ResultsPage() {
  const { runId } = useParams();
  const [loading, setLoading] = useState(true);
  const [run, setRun] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchRun = async () => {
      try {
        setLoading(true);
        const res = await getRun(runId);
        setRun(res);
      } catch (e) {
        setError(e.message || 'Failed to fetch run results.');
      } finally {
        setLoading(false);
      }
    };

    fetchRun();
  }, [runId]);

  if (loading) {
    return (
      <div className="text-center py-20 text-slate-500">
        ⏳ Loading results...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-20 text-red-500">
        ⚠️ {error}
        <div className="mt-6">
          <Link
            to="/upload"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Try Again
          </Link>
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="text-center py-20 text-slate-500">
        No results available.
      </div>
    );
  }

  // ✅ Construct proper download URL from backend
  const base = process.env.REACT_APP_API_URL || 'http://localhost:8001';
  const downloadUrl = `${base}/api/paper/download/${run.id}`;

  return (
    <div className="mt-10">
      <h2 className="text-2xl font-semibold tracking-tight mb-4">
        Paper Generation Results
      </h2>

      <div className="bg-white shadow rounded-xl p-6">
        <p>
          <strong>Run ID:</strong> {run.id}
        </p>
        <p>
          <strong>Status:</strong>{' '}
          <span
            className={`${
              run.status === 'completed' ? 'text-green-600' : 'text-yellow-600'
            }`}
          >
            {run.status}
          </span>
        </p>

        {run.status === 'completed' && (
          <div className="mt-4">
            <a
              href={downloadUrl}
              className="px-4 py-2 bg-green-600 text-white rounded-lg shadow hover:bg-green-700"
              download
            >
              ⬇️ Download Paper
            </a>
          </div>
        )}
      </div>

      {run.status === 'completed' && (
        <div className="mt-10">
          <PaperPreview url={downloadUrl} />
        </div>
      )}

      <div className="mt-10">
        <Link
          to="/upload"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Upload Another Notebook
        </Link>
      </div>
    </div>
  );
}
