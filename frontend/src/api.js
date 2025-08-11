import axios from "axios";

// Correct API base URL pointing to Flask backend on port 5000
const API_BASE = "http://127.0.0.1:5000";

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  
  // Use the API_BASE URL here (Flask backend)
  const res = await axios.post(`${API_BASE}/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const generatePaper = async (filePath, sections) => {
  const res = await axios.post(`${API_BASE}/generate`, {
    file_path: filePath,
    sections: sections,
  });
  return res.data;
};

export const downloadPDF = (pdfPath) => {
  // Opens the PDF download link in a new tab
  window.open(`${API_BASE}/download?path=${encodeURIComponent(pdfPath)}`, "_blank");
};
