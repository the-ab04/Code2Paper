import React, { useState } from "react";
import { uploadFile } from "../api";

export default function FileUpload({ onUpload }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");
    setLoading(true);
    try {
      const data = await uploadFile(file);
      onUpload(data.file_path);
    } catch (err) {
      alert("Upload failed");
    }
    setLoading(false);
  };

  return (
    <div className="bg-white p-6 rounded shadow w-full max-w-md mx-auto">
      <h2 className="text-xl font-semibold mb-4">1️⃣ Upload ML Code</h2>
      <input
        type="file"
        accept=".py,.ipynb"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-4"
      />
      <button
        onClick={handleUpload}
        disabled={loading}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        {loading ? "Uploading..." : "Upload"}
      </button>
    </div>
  );
}

