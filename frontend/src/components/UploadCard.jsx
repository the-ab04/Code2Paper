import { useState, useRef } from 'react';

const DEFAULT_SECTIONS = [
  'title',
  'abstract',
  'introduction',
  'methods',
  'experiments',
  'results',
  'discussion',
  'conclusion',
];

export default function UploadCard({ onSubmit, loading }) {
  const [fileName, setFileName] = useState('');
  const [title, setTitle] = useState('Auto-Generated Paper');
  const [author, setAuthor] = useState('Anonymous');
  const [selectedSections, setSelectedSections] = useState([...DEFAULT_SECTIONS]); // all selected by default
  const [selectAll, setSelectAll] = useState(true);

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

  const toggleSection = (section) => {
    setSelectedSections((prev) => {
      if (prev.includes(section)) {
        const next = prev.filter((s) => s !== section);
        setSelectAll(next.length === DEFAULT_SECTIONS.length);
        return next;
      } else {
        const next = [...prev, section];
        setSelectAll(next.length === DEFAULT_SECTIONS.length);
        return next;
      }
    });
  };

  const toggleSelectAll = () => {
    if (selectAll) {
      setSelectedSections([]);
      setSelectAll(false);
    } else {
      setSelectedSections([...DEFAULT_SECTIONS]);
      setSelectAll(true);
    }
  };

  const handleSubmit = async () => {
    const f = inputRef.current?.files?.[0];
    if (!f) {
      alert('Select a .ipynb file first');
      return;
    }

    // Ensure at least one section
    const sectionsToSend = selectedSections.length ? selectedSections : [...DEFAULT_SECTIONS];

    const fd = new FormData();
    fd.append('file', f); // matches backend parameter name
    fd.append('title', title);
    fd.append('author', author);

    try {
      // onSubmit now receives FormData and an array of sections to generate
      await onSubmit(fd, sectionsToSend);
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

      {/* Section Selection */}
      <div className="mt-6">
        <label className="block text-sm text-slate-300 mb-2">Choose sections to generate</label>
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={toggleSelectAll}
            type="button"
            className="px-3 py-1 rounded-md bg-slate-800/40 text-sm text-slate-200 border border-slate-700/50 hover:bg-slate-800/60"
          >
            {selectAll ? 'Unselect All' : 'Select All'}
          </button>
          <span className="text-slate-400 text-sm">{selectedSections.length} selected</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {DEFAULT_SECTIONS.map((s) => (
            <label
              key={s}
              className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer border ${
                selectedSections.includes(s)
                  ? 'border-indigo-500 bg-indigo-600/10 text-white'
                  : 'border-slate-700/30 text-slate-300'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedSections.includes(s)}
                onChange={() => toggleSection(s)}
                className="w-4 h-4"
              />
              <span className="capitalize text-sm">{s}</span>
            </label>
          ))}
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
