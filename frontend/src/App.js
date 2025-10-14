import { useState } from 'react';
import Header from './components/Header';
import UploadCard from './components/UploadCard';
import PaperPreview from './components/PaperPreview';
import { uploadNotebook, generatePaper } from './api/client';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [fileUrl, setFileUrl] = useState('');

  const handleUploadAndGenerate = async (formData) => {
    try {
      setLoading(true);

      // === Step 1: Upload notebook → backend returns run info ===
      const uploadRes = await uploadNotebook(formData);
      if (!uploadRes?.id) {
        throw new Error('Upload failed: No run_id returned.');
      }
      const runId = uploadRes.id;

      // === Step 2: Generate paper → backend returns { run_id, download_url } ===
      const genRes = await generatePaper(runId);
      if (genRes?.download_url) {
        // ✅ Ensure base URL comes from .env or fallback localhost
        const base = process.env.REACT_APP_API_URL || 'http://localhost:8001';
        setFileUrl(`${base}${genRes.download_url}`);
      } else {
        throw new Error('No download URL received from server.');
      }
    } catch (e) {
      console.error('Error in handleUploadAndGenerate:', e);
      alert('Error: ' + (e?.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* === Header === */}
      <Header />

      {/* === Main Content === */}
      <main className="flex-grow max-w-6xl mx-auto px-6 pb-20">
        <section className="mt-6 md:mt-10">
          <div className="mb-6">
            <h2 className="text-2xl md:text-3xl font-semibold tracking-tight">
              Turn Notebooks into Papers
            </h2>
            <p className="text-slate-400 mt-2">
              Upload a <code>.ipynb</code> notebook or script. We’ll parse facts,
              generate sections with Groq, enrich citations, and export a polished DOCX.
            </p>
          </div>

          {/* === Upload & Generate === */}
          <UploadCard onSubmit={handleUploadAndGenerate} loading={loading} />

          {/* === Download Preview === */}
          <PaperPreview url={fileUrl} />
        </section>
      </main>

      {/* === Footer === */}
      <footer className="text-center text-xs text-slate-500 py-8">
        © {new Date().getFullYear()} Code2Paper
      </footer>
    </div>
  );
}
