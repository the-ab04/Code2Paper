// src/api/client.js
import axios from "axios";

// Use REACT_APP_API_URL from .env or default to backend local URL
const base = process.env.REACT_APP_API_URL || "http://localhost:8001";

const api = axios.create({
  baseURL: base,
  timeout: 60000, // 60s timeout
});

// --- Helpers to extract useful error messages ---
function _extractErrorMessage(error) {
  // Try to pick the most useful message possible
  if (!error) return "Unknown error";
  if (error.response) {
    // Prefer detail->message or detail keys often used by FastAPI
    const data = error.response.data;
    if (data) {
      if (typeof data === "string") return data;
      // common FastAPI shape: { detail: "..." }
      if (data.detail) return data.detail;
      // sometimes nested { error: { message: "..." } }
      if (data.error && data.error.message) return data.error.message;
      // fallback to JSON string
      try {
        return JSON.stringify(data);
      } catch (e) {
        return String(data);
      }
    }
    return `${error.response.status} ${error.response.statusText}`;
  }
  if (error.message) return error.message;
  return String(error);
}

/**
 * Upload a notebook (.ipynb) → creates a Run in DB
 * @param {FormData} formData
 * @returns {Promise<Object>} Run object { id, status, input_file, ... }
 */
export async function uploadNotebook(formData) {
  try {
    const res = await api.post("/api/paper/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      // You can add onUploadProgress here if you want fine-grained upload progress
    });
    return res.data;
  } catch (error) {
    console.error("Upload Error:", error.response?.data || error.message);
    throw new Error(_extractErrorMessage(error) || "Failed to upload notebook");
  }
}

/**
 * Generate paper from an existing run → returns generation response (run_id, download_url, produced_sections...)
 * Accepts an optional body param for passing `{ sections: [...], use_rag: true }`.
 * Backwards compatible: call with generatePaper(runId) and no body.
 *
 * @param {number|string} runId
 * @param {Object} [body={}] optional payload forwarded to backend
 * @returns {Promise<Object>}
 */
export async function generatePaper(runId, body = {}) {
  try {
    // default body if omitted — backend may have defaults
    const payload = body || {};
    const res = await api.post(`/api/paper/generate/${runId}`, payload);
    return res.data;
  } catch (error) {
    console.error("Generate Paper Error:", error.response?.data || error.message);
    throw new Error(_extractErrorMessage(error) || "Failed to generate paper");
  }
}

/**
 * Explicit alias that makes intent clear in the frontend code.
 * @param {number|string} runId
 * @param {Object} body
 * @returns {Promise<Object>}
 */
export async function generatePaperWithBody(runId, body) {
  return generatePaper(runId, body);
}

/**
 * Download a generated paper (DOCX) → returns raw Blob
 * @param {number|string} runId
 * @returns {Promise<Blob>}
 */
export async function downloadPaper(runId) {
  try {
    const res = await api.get(`/api/paper/download/${runId}`, {
      responseType: "blob",
    });
    return res.data;
  } catch (error) {
    console.error("Download Error:", error.response?.data || error.message);
    throw new Error(_extractErrorMessage(error) || "Failed to download paper");
  }
}

/**
 * Fetch metadata of a specific run
 * @param {number|string} runId
 * @returns {Promise<Object>}
 */
export async function getRun(runId) {
  try {
    const res = await api.get(`/api/paper/runs/${runId}`);
    return res.data;
  } catch (error) {
    console.error("Get Run Error:", error.response?.data || error.message);
    throw new Error(_extractErrorMessage(error) || "Failed to fetch run");
  }
}

/**
 * Fetch list of all runs
 * @returns {Promise<Array>}
 */
export async function listRuns() {
  try {
    const res = await api.get("/api/paper/runs");
    return res.data;
  } catch (error) {
    console.error("List Runs Error:", error.response?.data || error.message);
    throw new Error(_extractErrorMessage(error) || "Failed to fetch runs");
  }
}

export default api;
