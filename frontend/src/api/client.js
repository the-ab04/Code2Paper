import axios from 'axios';

// ✅ Use REACT_APP_API_URL from .env or default to backend local URL
const base = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: base,
  timeout: 300000, // 60s timeout
});

/**
 * Upload a notebook (.ipynb) → creates a Run in DB
 * @param {FormData} formData
 * @returns {Promise<Object>} Run object { id, status, input_file, ... }
 */
export async function uploadNotebook(formData) {
  try {
    const res = await api.post('/api/paper/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  } catch (error) {
    console.error('Upload Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.detail || 'Failed to upload notebook');
  }
}

/**
 * Generate paper from an existing run → returns { run_id, download_url, produced_sections }
 * @param {number|string} runId
 * @param {Array<string>|null} sections Optional list of sections to generate (e.g. ["abstract","methods"])
 * @param {boolean} use_rag Whether to include RAG retrieval when generating (default true)
 * @returns {Promise<Object>}
 */
export async function generatePaper(runId, sections = null, use_rag = true) {
  try {
    const payload = {
      // if sections is null, backend will generate all sections
      sections: sections,
      use_rag: use_rag,
    };

    const res = await api.post(`/api/paper/generate/${runId}`, payload, {
      headers: { 'Content-Type': 'application/json' },
    });

    return res.data;
  } catch (error) {
    console.error('Generate Paper Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.detail || 'Failed to generate paper');
  }
}

/**
 * Download a generated paper (DOCX) → returns Blob
 * @param {number|string} runId
 * @returns {Promise<Blob>}
 */
export async function downloadPaper(runId) {
  try {
    const res = await api.get(`/api/paper/download/${runId}`, { responseType: 'blob' });
    return res.data;
  } catch (error) {
    console.error('Download Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.detail || 'Failed to download paper');
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
    console.error('Get Run Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.detail || 'Failed to fetch run');
  }
}

/**
 * Fetch list of all runs
 * @returns {Promise<Array>}
 */
export async function listRuns() {
  try {
    const res = await api.get('/api/paper/runs');
    return res.data;
  } catch (error) {
    console.error('List Runs Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.detail || 'Failed to fetch runs');
  }
}

export default api;
