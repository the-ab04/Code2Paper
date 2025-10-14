import { useState } from 'react';
import Header from '../components/Header';
import UploadCard from '../components/UploadCard';
import PaperPreview from '../components/PaperPreview';
import { uploadNotebook, generatePaper } from '../api/client';

export default function UploadPage() {
  const [loading, setLoading] = useState(false);
  const [fileUrl, setFileUrl] = useState('');

  const handleUploadAndGenerate = async (fd) => {
    try {
      setLoading(true);

      // === Step 1: Upload notebook → get run_id ===
      const uploadRes = await uploadNotebook(fd);
      if (!uploadRes?.id) {
        throw new Error('Upload failed: No run_id returned.');
      }
      const runId = uploadRes.id;

      // === Step 2: Generate paper → backend returns { run_id, download_url } ===
      const genRes = await generatePaper(runId);
      if (!genRes?.download_url) {
        throw new Error('No download URL received from server.');
      }

      // Build final file URL
      const base = process.env.REACT_APP_API_URL || 'http://localhost:8001';
      setFileUrl(`${base}${genRes.download_url}`);
    } catch (e) {
      alert('Error: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Header />
      <main className="max-w-6xl mx-auto px-6 pb-20">
        <section className="mt-6 md:mt-10">
          <div className="mb-6">
            <h2 className="text-2xl md:text-3xl font-semibold tracking-tight">
              Upload Notebook → Generate Paper
            </h2>
            <p className="text-slate-400 mt-2">
              Upload a <code>.ipynb</code> notebook or <code>.py</code> script.
              Code2Paper will parse it, generate structured sections with citations,
              and export a polished DOCX paper you can download.
            </p>
          </div>

          <UploadCard onSubmit={handleUploadAndGenerate} loading={loading} />

          {fileUrl && <PaperPreview url={fileUrl} />}
        </section>
      </main>

      <footer className="text-center text-xs text-slate-500 py-8">
        © {new Date().getFullYear()} Code2Paper
      </footer>
    </div>
  );
}
