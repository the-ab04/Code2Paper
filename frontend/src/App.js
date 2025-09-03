import { useState } from 'react';
import Header from './components/Header';
import UploadCard from './components/UploadCard';
import PaperPreview from './components/PaperPreview';
import { generatePaper } from './api/client';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState('');

  const handleGenerate = async (fd) => {
    try {
      setLoading(true);
      const res = await generatePaper(fd);

      if (res.job_id) {
        // Use CRA env variable instead of import.meta
        const base = process.env.REACT_APP_API_URL || 'http://localhost:8001';
        setPdfUrl(`${base}/paper/download/${res.job_id}`);
      } else if (res.pdf_path) {
        setPdfUrl(res.pdf_path);
      }
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
            <h2 className="text-2xl md:text-3xl font-semibold tracking-tight">Turn Notebooks into Papers</h2>
            <p className="text-slate-400 mt-2">
              Upload a .ipynb notebook. We’ll parse facts, write sections with Groq, add citations, and export a LaTeX-quality PDF.
            </p>
          </div>
          <UploadCard onSubmit={handleGenerate} loading={loading} />
          <PaperPreview url={pdfUrl} />
        </section>
      </main>
      <footer className="text-center text-xs text-slate-500 py-8">© {new Date().getFullYear()} Code2Paper</footer>
    </div>
  );
}
