import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001/api/paper', // ✅ Backend route
  timeout: 60000,
});

/**
 * Generate paper and download as DOCX
 * @param {FormData} formData
 */
export async function generatePaper(formData) {
  try {
    const res = await api.post('/generate', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob', // ✅ Important for downloading file
    });

    // ✅ Create download link dynamically
    const contentDisposition = res.headers['content-disposition'];
    let filename = 'code2paper_output.docx'; // Default filename
    if (contentDisposition) {
      const match = contentDisposition.match(/filename="?(.+)"?/);
      if (match) filename = match[1];
    }

    const blob = new Blob([res.data], {
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    });
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);

    return true;
  } catch (error) {
    console.error('Generate Paper Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.detail || 'Failed to generate paper');
  }
}

export default api;
