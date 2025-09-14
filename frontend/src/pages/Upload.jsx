import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [serverHint, setServerHint] = useState(null);
  const [lastError, setLastError] = useState(null);
  const navigate = useNavigate();

  // Try to learn the expected form field from FastAPI's openapi if available
  useEffect(() => {
    async function probe() {
      try {
        const res = await axios.get(`${API}/openapi.json`, { timeout: 3000 });
        const spec = res.data;
        const p = spec?.paths?.["/upload"];
        const req = p?.post?.requestBody?.content?.["multipart/form-data"]?.schema;
        const props = req?.properties || {};
        const required = p?.post?.requestBody?.required ? (req?.required || []) : [];
        const keys = Object.keys(props);
        if (keys.length > 0) {
          setServerHint({ keys, required });
        }
      } catch {}
    }
    probe();
  }, []);

  async function tryUploadWithField(fieldName) {
    const fd = new FormData();
    fd.append(fieldName, file);
    // IMPORTANT: don't set Content-Type manually; let the browser add the boundary
    const res = await axios.post(`${API}/upload`, fd, {
      onUploadProgress: (evt) => {
        if (evt.total) {
          setProgress(Math.round((evt.loaded * 100) / evt.total));
        }
      },
      timeout: 120000,
    });
    return res;
  }

  async function onSubmit(e) {
    e.preventDefault();
    setLastError(null);
    if (!file) return alert("請先選擇影片檔案");
    setBusy(true);
    setProgress(0);

    // Candidate field names (common in FastAPI examples)
    let candidates = ["file", "video", "video_file", "upload_file"];
    if (serverHint?.keys?.length) {
      // Put server-advertised keys first
      const serverKeys = serverHint.keys;
      candidates = [...serverKeys, ...candidates.filter(k => !serverKeys.includes(k))];
    }

    try {
      let ok = false;
      let lastErr = null;
      for (const name of candidates) {
        try {
          const res = await tryUploadWithField(name);
          ok = true;
          break;
        } catch (err) {
          lastErr = err;
          // Only continue fallback on 422 (validation)
          const status = err?.response?.status;
          if (status !== 422) {
            throw err;
          }
        }
      }
      if (!ok) {
        throw lastErr || new Error("Unknown upload error");
      }
      navigate("/results");
    } catch (err) {
      let details = "";
      const r = err?.response;
      if (r?.data) {
        try {
          details = JSON.stringify(r.data, null, 2);
        } catch {
          details = String(r.data);
        }
      } else {
        details = err.message || String(err);
      }
      setLastError(details);
      alert("上傳失敗：\n" + details);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">上傳比賽影片</h2>
      {serverHint && (
        <div className="mb-3 text-xs p-2 rounded bg-amber-50 border border-amber-200">
          偵測到後端可能接受的欄位：
          <code className="ml-1">{serverHint.keys.join(", ")}</code>
          {serverHint.required?.length ? (
            <span>（必填：{serverHint.required.join(", ")}）</span>
          ) : null}
        </div>
      )}
      <form onSubmit={onSubmit} className="space-y-4 max-w-xl">
        <input
          type="file"
          name="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        {busy && (
          <div className="text-sm">
            上傳中... {progress}%
            <div className="h-2 bg-neutral-200 rounded mt-1 overflow-hidden">
              <div
                className="h-full bg-black"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
        <button
          disabled={busy || !file}
          className="px-4 py-2 rounded border bg-black text-white disabled:opacity-60"
        >
          {busy ? "上傳中..." : "開始上傳"}
        </button>
      </form>

      {lastError && (
        <pre className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-xs overflow-auto">
{lastError}
        </pre>
      )}
    </div>
  );
}
