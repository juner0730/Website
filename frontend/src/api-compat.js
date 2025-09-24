// src/api-compat.js
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, { credentials: "include", ...opts });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : null;
}

// 兼容 onepage 直接呼叫的全域函式：apiListVideos()
// 先嘗試新端點 /api/videos；失敗就退回舊端點 /videos
export async function apiListVideos() {
  try {
    return await fetchJSON(`${API_BASE}/api/videos`);
  } catch {
    return await fetchJSON(`${API_BASE}/videos`);
  }
}

// 若 onepage 之後要用到上傳，順便提供 apiUpload(file)
export async function apiUpload(file) {
  const form = new FormData();
  form.append("file", file);
  // 同樣支援新舊兩種路徑
  try {
    return await fetchJSON(`${API_BASE}/api/upload`, { method: "POST", body: form });
  } catch {
    return await fetchJSON(`${API_BASE}/upload`, { method: "POST", body: form });
  }
}

// 掛到 window，讓 onepage 直接用（不需要 import）
window.apiListVideos = apiListVideos;
window.apiUpload = apiUpload;
