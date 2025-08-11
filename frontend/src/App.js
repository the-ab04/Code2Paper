import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import SectionSelector from "./components/SectionSelector";
import OutputPanel from "./components/OutputPanel";
import { generatePaper } from "./api";

function App() {
  const [filePath, setFilePath] = useState(null);
  const [pdfPath, setPdfPath] = useState(null);

  const handleGenerate = async (sections) => {
    if (!filePath) return alert("Please upload a file first");
    const data = await generatePaper(filePath, sections);
    if (data.pdf_path) {
      setPdfPath(data.pdf_path);
    } else {
      alert("Generation failed");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-center mb-8">🧠 Code2Paper</h1>
      <FileUpload onUpload={setFilePath} />
      {filePath && <SectionSelector onGenerate={handleGenerate} />}
      <OutputPanel pdfPath={pdfPath} />
    </div>
  );
}

export default App;
