import React from "react";
import { downloadPDF } from "../api";

export default function OutputPanel({ pdfPath }) {
  if (!pdfPath) return null;

  return (
    <div className="bg-white p-6 rounded shadow w-full max-w-md mx-auto mt-6 text-center">
      <h2 className="text-xl font-semibold mb-4">3️⃣ Paper Ready</h2>
      <button
        onClick={() => downloadPDF(pdfPath)}
        className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
      >
        Download PDF
      </button>
    </div>
  );
}
