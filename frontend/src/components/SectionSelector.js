import React, { useState } from "react";

const allSections = ["abstract", "methodology", "results", "conclusion"];

export default function SectionSelector({ onGenerate }) {
  const [selected, setSelected] = useState([]);

  const toggleSection = (sec) => {
    setSelected((prev) =>
      prev.includes(sec) ? prev.filter((s) => s !== sec) : [...prev, sec]
    );
  };

  const toggleSelectAll = () => {
    if (selected.length === allSections.length) {
      // If all selected, deselect all
      setSelected([]);
    } else {
      // Select all sections
      setSelected(allSections);
    }
  };

  return (
    <div className="bg-white p-6 rounded shadow w-full max-w-md mx-auto mt-6">
      <h2 className="text-xl font-semibold mb-4">2️⃣ Select Sections</h2>

      <button
        onClick={toggleSelectAll}
        className="mb-4 px-4 py-2 rounded border bg-yellow-400 text-black hover:bg-yellow-500"
      >
        {selected.length === allSections.length ? "Deselect All" : "Select All"}
      </button>

      <div className="grid grid-cols-2 gap-2 mb-4">
        {allSections.map((sec) => (
          <button
            key={sec}
            onClick={() => toggleSection(sec)}
            className={`px-3 py-2 rounded border ${
              selected.includes(sec)
                ? "bg-green-500 text-white"
                : "bg-gray-100"
            }`}
          >
            {sec}
          </button>
        ))}
      </div>

      <button
        onClick={() => onGenerate(selected)}
        disabled={selected.length === 0}
        className={`px-4 py-2 rounded text-white ${
          selected.length === 0
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-blue-500 hover:bg-blue-600"
        }`}
      >
        Generate Paper
      </button>
    </div>
  );
}
