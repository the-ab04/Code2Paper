// src/components/PaperPreview.jsx
import React from 'react';

export default function PaperPreview({ url }) {
  if (!url) {
    return (
      <div className="mt-6">
        <p className="text-slate-500">No generated paper yet.</p>
      </div>
    );
  }

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold">Generated Paper</h3>
      <p className="text-slate-400 text-sm mb-3">Click to download the generated DOCX.</p>
      <a
        href={url}
        className="inline-block px-4 py-2 bg-green-600 text-white rounded-lg"
        target="_blank"
        rel="noreferrer"
        download
      >
        Download Paper
      </a>
    </div>
  );
}
