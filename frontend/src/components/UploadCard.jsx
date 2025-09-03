import { useState, useRef } from 'react';

export default function UploadCard({ onSubmit, loading }) {
  const [fileName, setFileName] = useState('');
  const [title, setTitle] = useState('Auto-Generated Paper');
  const [author, setAuthor] = useState('Anonymous');
  const inputRef = useRef();

  const onDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    if (!f.name.endsWith('.ipynb')) {
      alert('Please upload a .ipynb file');
      return;
    }
    inputRef.current.files = e.dataTransfer.files;
    setFileName(f.name);
  };

  const pick = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.name.endsWith('.ipynb')) {
      alert('Please upload a .ipynb file');
      return;
    }
    setFileName(f.name);
  };

  const handleSubmit = async () => {
    const f = inputRef.current?.files?.[0];
    if (!f) {
      alert('Select a .ipynb file first');
      return;
    }

    const fd = new FormData();
    fd.append('nb_file', f); // ✅ Match backend parameter
    fd.append('style', 'ieee');
    fd.append('title', title);
    fd.append('author', author);

    try {
      await onSubmit(fd);
    } catch (err) {
      console.error('Upload error:', err);
      alert(`Error: ${err.message || 'Failed to generate paper'}`);
    }
  };

  return (
    <div className="glass rounded-2xl p-6 md:p-8">
      {/* Drag & Drop Area */}
      <div
        className="border-2 border-dashed border-slate-500/40 rounded-xl p-8 grid place-items-center text-center hover:border-indigo-400/60 transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
      >
        <input ref={inputRef} hidden type="file" accept=".ipynb" onChange={pick} />
        <div className="space-y-3">
          <p className="text-slate-300">
            Drag & drop your <span className="text-white font-medium">.ipynb</span> here
          </p>
          <p className="text-slate-500 text-sm">or</p>
          <button
            onClick={() => inputRef.current?.click()}
            className="px-4 py-2 rounded-xl bg-indigo-500/20 text-indigo-200 border border-indigo-400/30 hover:bg-indigo-500/30"
          >
            Browse file
          </button>
          {fileName && <p className="text-xs text-slate-400">Selected: {fileName}</p>}
        </div>
      </div>

      {/* Title and Author Inputs */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-slate-300 mb-1">Title</label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full rounded-lg bg-slate-900/40 border border-slate-700/50 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400/30"
            placeholder="Paper title"
          />
        </div>
        <div>
          <label className="block text-sm text-slate-300 mb-1">Author</label>
          <input
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            className="w-full rounded-lg bg-slate-900/40 border border-slate-700/50 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400/30"
            placeholder="Your name"
          />
        </div>
      </div>

      {/* Submit Button */}
      <div className="mt-6 flex items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`px-5 py-2.5 rounded-xl bg-indigo-500 text-white shadow-glow hover:brightness-110 disabled:opacity-50 ${
            loading ? 'cursor-not-allowed' : ''
          }`}
        >
          {loading ? 'Generating…' : 'Generate Paper'}
        </button>
        <span className="text-slate-400 text-sm">Only .ipynb supported</span>
      </div>
    </div>
  );
}
