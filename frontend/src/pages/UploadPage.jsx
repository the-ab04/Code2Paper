// src/pages/UploadPage.jsx
import React, { useRef, useState, useEffect } from "react";
import "../styles/Uploadpage.css";
import api, { uploadNotebook } from "../api/client"; // axios client + helper

export default function UploadPage() {
  const uploadSectionRef = useRef(null);
  const fileInputRef = useRef(null);
  const progressTimerRef = useRef(null);

  const [uploadedFile, setUploadedFile] = useState(null);
  const [runId, setRunId] = useState(null); // id returned by upload
  const [selectedSections, setSelectedSections] = useState(new Set());
  const [progress, setProgress] = useState(0); // 0..100
  const [generating, setGenerating] = useState(false);
  const [statusText, setStatusText] = useState("Idle");

  const frontendSections = [
    "abstract",
    "introduction",
    "literature_review", // NEW: literature review
    "methodology", // maps to backend 'methods'
    "results",
    "conclusion",
    "references",
  ];

  const canonicalMap = {
    abstract: "abstract",
    introduction: "introduction",
    literature_review: "literature_review", // make sure backend accepts this
    methodology: "methods",
    results: "results",
    conclusion: "conclusion",
    references: "references",
  };

  useEffect(() => {
    const el = uploadSectionRef.current;
    if (!el) return;

    const onDragOver = (e) => {
      e.preventDefault();
      el.classList.add("dragover");
    };
    const onDragLeave = () => el.classList.remove("dragover");
    const onDrop = (e) => {
      e.preventDefault();
      el.classList.remove("dragover");
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    };

    el.addEventListener("dragover", onDragOver);
    el.addEventListener("dragleave", onDragLeave);
    el.addEventListener("drop", onDrop);

    return () => {
      el.removeEventListener("dragover", onDragOver);
      el.removeEventListener("dragleave", onDragLeave);
      el.removeEventListener("drop", onDrop);
    };
  }, []);

  // Validate that the file is .ipynb and looks like a Jupyter notebook
  async function handleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".ipynb")) {
      alert("Only .ipynb (Jupyter Notebook) files are supported.");
      return;
    }

    try {
      const text = await readFileAsText(file);
      let parsed;
      try {
        parsed = JSON.parse(text);
      } catch (err) {
        alert("The uploaded file is not valid JSON. Please upload a valid .ipynb file.");
        return;
      }

      // Basic notebook check
      if (!parsed || typeof parsed !== "object" || !("nbformat" in parsed)) {
        alert("This does not look like a valid Jupyter notebook (.ipynb).");
        return;
      }

      setUploadedFile(file);
      setStatusText(`Selected: ${file.name}`);
      // Clear prior runId since new upload will create a new run
      setRunId(null);
      setProgress(0);
    } catch (err) {
      console.error("Failed to read file:", err);
      alert("There was an error reading the file. Try again.");
    }
  }

  function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => {
        reader.abort();
        reject(new Error("Problem parsing input file."));
      };
      reader.onload = () => resolve(reader.result);
      reader.readAsText(file);
    });
  }

  function toggleSection(section) {
    setSelectedSections((prev) => {
      const next = new Set(prev);
      const combinedAll = frontendSections;
      if (section === "combined") {
        const allActive = combinedAll.every((s) => next.has(s));
        if (allActive) next.clear();
        else combinedAll.forEach((s) => next.add(s));
      } else {
        if (next.has(section)) next.delete(section);
        else next.add(section);
      }
      return next;
    });
  }

  // Smoothly increment progress while waiting for long tasks
  function startProgressAnimation(target = 85) {
    // Clear existing
    clearInterval(progressTimerRef.current);
    progressTimerRef.current = setInterval(() => {
      setProgress((p) => {
        // slowly approach target (but don't reach 100 here)
        if (p < target) {
          const delta = Math.max(1, Math.floor((target - p) / 12));
          return Math.min(target, p + delta);
        }
        return p;
      });
    }, 600); // every 600ms
  }

  function stopProgressAnimation(final = 100) {
    clearInterval(progressTimerRef.current);
    progressTimerRef.current = null;
    setProgress(final);
  }

  async function generatePaper() {
    if (!uploadedFile) {
      alert("Please upload a .ipynb file first!");
      return;
    }
    if (selectedSections.size === 0) {
      alert("Please select at least one section!");
      return;
    }

    // Map to backend canonical names and dedupe (exclude "combined" if present)
    const sectionsList = Array.from(selectedSections)
      .filter((s) => s !== "combined")
      .map((s) => canonicalMap[s])
      .filter(Boolean);

    setGenerating(true);
    setProgress(3);
    setStatusText("Uploading notebook...");

    // Start a progress animation during upload/generation
    startProgressAnimation(40);

    try {
      // Step A: Upload notebook -> create a run
      const fd = new FormData();
      fd.append("file", uploadedFile);

      // If your uploadNotebook helper supports onUploadProgress, you can pass it.
      // Here we just call uploadNotebook (which uses axios internally).
      const uploadRes = await uploadNotebook(fd);
      if (!uploadRes?.id) throw new Error("Upload failed: no run id returned");
      const newRunId = uploadRes.id;
      setRunId(newRunId);

      setStatusText("Notebook uploaded. Requesting generation...");
      // bump progress a bit
      setProgress((p) => Math.max(p, 30));
      // continue animated progress
      startProgressAnimation(70);

      // Step B: Trigger generation. Backend expects JSON body { sections, use_rag }.
      const body = { sections: sectionsList, use_rag: true };
      const genResp = await api.post(`/api/paper/generate/${newRunId}`, body, {
        timeout: 300000, // may take several minutes
      });

      const genData = genResp?.data;
      setStatusText("Server finished generation. Preparing download...");

      // Step C: Determine download URL
      let downloadUrl = genData?.download_url || `/api/paper/download/${newRunId}`;

      // Step D: Download the file (blob)
      const dlResp = await api.get(downloadUrl, {
        responseType: "blob",
        timeout: 300000,
      });

      // Extract filename
      const contentType =
        dlResp.headers["content-type"] ||
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
      const blob = new Blob([dlResp.data], { type: contentType });

      let filename = `generated_paper_${newRunId}.docx`;
      const cd = dlResp.headers["content-disposition"];
      if (cd) {
        const match = /filename\*?=(?:UTF-8'')?["']?([^;"']+)/i.exec(cd);
        if (match && match[1]) {
          try {
            filename = decodeURIComponent(match[1].replace(/['"]/g, ""));
          } catch (e) {
            filename = match[1].replace(/['"]/g, "");
          }
        }
      }

      // Stop animation and move to near-complete
      stopProgressAnimation(95);

      // Trigger browser download
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      // finalize
      stopProgressAnimation(100);
      setStatusText("✅ Paper generated and downloaded");
    } catch (err) {
      console.error("Generate error:", err);
      // show server error if available
      let msg = err?.message || "Generation failed";
      if (err?.response?.data) {
        msg = err.response.data.detail || JSON.stringify(err.response.data) || msg;
      }
      setStatusText("Error: " + msg);
      alert("Error during generation: " + msg);
      stopProgressAnimation(0);
    } finally {
      setGenerating(false);
    }
  }

  // Download an already-generated paper for the stored runId (if present).
  async function downloadPaper() {
    try {
      let targetRunId = runId;
      if (!targetRunId) {
        // fallback: query runs and pick last
        const runsRes = await api.get("/api/paper/runs");
        const runs = runsRes?.data || [];
        if (!runs.length) {
          alert("No runs found on server. Please generate first.");
          return;
        }
        targetRunId = runs[runs.length - 1].id;
      }

      setStatusText("⬇️ Downloading generated paper...");
      setGenerating(true);
      startProgressAnimation(40);

      const dlResp = await api.get(`/api/paper/download/${targetRunId}`, {
        responseType: "blob",
        timeout: 120000,
      });

      const contentType =
        dlResp.headers["content-type"] ||
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
      const blob = new Blob([dlResp.data], { type: contentType });

      let filename = `generated_paper_${targetRunId}.docx`;
      const cd = dlResp.headers["content-disposition"];
      if (cd) {
        const match = /filename\*?=(?:UTF-8'')?["']?([^;"']+)/i.exec(cd);
        if (match && match[1]) {
          try {
            filename = decodeURIComponent(match[1].replace(/['"]/g, ""));
          } catch (e) {
            filename = match[1].replace(/['"]/g, "");
          }
        }
      }

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      stopProgressAnimation(100);
      setStatusText("✅ Download complete");
    } catch (err) {
      console.error("Download error:", err);
      let msg = err?.message || "Download failed";
      if (err?.response?.data) {
        msg = err.response.data.detail || JSON.stringify(err.response.data) || msg;
      }
      setStatusText("Error: " + msg);
      alert("Error downloading paper: " + msg);
      stopProgressAnimation(0);
    } finally {
      setGenerating(false);
    }
  }

  function openFilePicker() {
    if (fileInputRef.current) fileInputRef.current.value = "";
    fileInputRef.current?.click();
  }

  return (
    <section className="upload-section-wrapper">
      <div className="floating-shapes" />
      <div className="container">
        <h1 className="page-title">GENERATE RESEARCH PAPER</h1>
        <p className="page-subtitle">
          Upload your Jupyter notebook (.ipynb) and transform it into a publication-ready paper
        </p>

        <div
          id="uploadSection"
          ref={uploadSectionRef}
          className={`upload-section ${uploadedFile ? "has-file" : ""}`}
        >
          <div className="upload-area">
            <div className="upload-icon">📄</div>
            <h3 className="upload-text">Drag & Drop your .ipynb file here</h3>
            <p className="upload-subtext">
              or click to browse (only <strong>.ipynb</strong> files supported)
            </p>

            <input
              ref={fileInputRef}
              type="file"
              id="fileInput"
              className="file-input"
              accept=".ipynb,application/x-ipynb+json,application/json"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              style={{ display: "none" }}
            />

            <button className="upload-btn" onClick={openFilePicker}>
              📁 Choose File
            </button>

            <div className={`file-info ${uploadedFile ? "active" : ""}`} id="fileInfo">
              <p>
                {uploadedFile ? (
                  <>
                    ✅ File uploaded:{" "}
                    <span className="file-name" id="fileName">
                      {uploadedFile.name}
                    </span>
                  </>
                ) : (
                  "No file selected"
                )}
              </p>
            </div>
          </div>
        </div>

        <div className="sections-container">
          <h2 className="section-title">SELECT SECTIONS TO GENERATE</h2>
          <div className="section-buttons">
            {frontendSections.map((s) => (
              <button
                key={s}
                className={`section-btn ${selectedSections.has(s) ? "active" : ""}`}
                onClick={() => toggleSection(s)}
                data-section={s}
                type="button"
              >
                {s === "abstract"
                  ? "📋 Abstract"
                  : s === "introduction"
                  ? "📖 Introduction"
                  : s === "literature_review"
                  ? "📚 Literature Review"
                  : s === "methodology"
                  ? "🔬 Methodology"
                  : s === "results"
                  ? "📊 Results"
                  : s === "discussion"
                  ? "💬 Discussion"
                  : s === "conclusion"
                  ? "✅ Conclusion"
                  : "📚 References"}
              </button>
            ))}

            <button
              className={`section-btn ${frontendSections.every((s) => selectedSections.has(s)) ? "active" : ""}`}
              data-section="combined"
              onClick={() => toggleSection("combined")}
              type="button"
            >
              🎯 Combined (All)
            </button>
          </div>

          <div className="action-buttons">
            <button
              className="action-btn generate-btn"
              id="generateBtn"
              onClick={generatePaper}
              disabled={generating}
              type="button"
            >
              ⚡ Generate Paper
            </button>

            <button
              className="action-btn download-btn"
              id="downloadBtn"
              onClick={downloadPaper}
              disabled={generating || progress < 100}
              type="button"
            >
              💾 Download Paper
            </button>
          </div>

          <div
            className={`progress-section ${generating || progress >= 100 ? "active" : ""}`}
            id="progressSection"
            aria-live="polite"
          >
            <div className="progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
              <div className="progress-fill" id="progressFill" style={{ width: `${progress}%` }} />
            </div>
            <p className="progress-text" id="progressText">
              {statusText}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
