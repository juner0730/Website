import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  UploadCloud,
  Loader2,
  CheckCircle2,
  XCircle,
  Film,
  Download,
  Search,
  Play,
  ChevronRight,
  Clock,
  AlertTriangle,
  Trash2,
} from "lucide-react";

/**
 * \u25CF One-Page Highlights Studio
 *
 * 功能 (全在單頁):
 * 1) 上傳原始影片 (頂部) \n
 * 2) 下方列出「歷史影片」(已上傳/處理中/已完成/失敗) \n
 * 3) 點選某筆「已完成」影片 => 顯示「自動剪輯精華」(依背號分組), 可即時播放或下載
 *
 * 後端 API (可依你實作調整路徑):
 *  - POST   /api/videos                         -> 上傳影片 (multipart/form-data: file)
 *  - GET    /api/videos?limit=50                -> 取最近列表
 *  - GET    /api/videos/:id                     -> 取單筆影片資訊 (含 status)
 *  - GET    /api/videos/:id/highlights          -> 取該影片的精華清單 (群組: jerseyNo)
 *  - GET    /api/videos/:id/highlights/archive  -> (可選) 全部精華的 zip 下載連結
 *  - DELETE /api/videos/:id                     -> (可選) 刪除影片與產物
 *
 * .env: VITE_API_BASE (預設 http://localhost:8000)
 */

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const fmtBytes = (bytes) => {
  if (!Number.isFinite(bytes)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024; i += 1;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
};

const fmtDuration = (sec) => {
  if (!sec && sec !== 0) return "-";
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);
  return [h, m, s].map((n) => String(n).padStart(2, "0")).join(":");
};

const timeAgo = (iso) => {
  if (!iso) return "-";
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}秒前`;
  if (diff < 3600) return `${Math.floor(diff/60)}分鐘前`;
  if (diff < 86400) return `${Math.floor(diff/3600)}小時前`;
  return `${Math.floor(diff/86400)}天前`;
};

const badgeClass = (status) => {
  switch (status) {
    case "queued":
      return "bg-amber-50 text-amber-700 border-amber-200";
    case "processing":
      return "bg-blue-50 text-blue-700 border-blue-200";
    case "done":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "failed":
      return "bg-rose-50 text-rose-700 border-rose-200";
    default:
      return "bg-neutral-50 text-neutral-700 border-neutral-200";
  }
};

const StatusBadge = ({ status }) => (
  <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs ${badgeClass(status)}`}>
    {status === "processing" && <Loader2 className="size-3 animate-spin" />} 
    {status === "done" && <CheckCircle2 className="size-3" />} 
    {status === "failed" && <XCircle className="size-3" />} 
    {status}
  </span>
);

/** 上傳：用 XHR 追蹤進度 (fetch 無上傳進度事件) */
async function uploadFileXHR(file, { onProgress }) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/videos`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress({ loaded: e.loaded, total: e.total, pct: Math.round((e.loaded / e.total) * 100) });
      }
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        try {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(xhr.responseText || `Upload failed: ${xhr.status}`));
          }
        } catch (e) {
          reject(e);
        }
      }
    };

    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

const useVideos = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/videos?limit=100`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setItems(Array.isArray(data) ? data : data.items || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);
  return { items, setItems, loading, error, reload: load };
};

const pollVideo = async (id, { signal } = {}) => {
  const res = await fetch(`${API_BASE}/api/videos/${id}`, { signal });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

const fetchHighlights = async (id) => {
  const res = await fetch(`${API_BASE}/api/videos/${id}/highlights`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

export default function OnePageHighlightsStudio() {
  const { items, setItems, loading, error, reload } = useVideos();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null); // video object
  const [highlights, setHighlights] = useState(null); // { jerseyNo: Clip[] }
  const [hlLoading, setHlLoading] = useState(false);
  const [hlError, setHlError] = useState(null);

  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const inputRef = useRef(null);

  // 本地快篩搜尋
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((v) =>
      [v.filename, v.title, v.id].filter(Boolean).some(s => String(s).toLowerCase().includes(q))
    );
  }, [items, query]);

  // 點選影片 => 如果是 done 才載入 highlights
  useEffect(() => {
    const loadHL = async () => {
      if (!selected) return;
      setHlError(null); setHighlights(null);
      if (selected.status !== "done") return;
      setHlLoading(true);
      try {
        const data = await fetchHighlights(selected.id);
        setHighlights(data?.groups || data || {});
      } catch (e) {
        setHlError(e.message);
      } finally {
        setHlLoading(false);
      }
    };
    loadHL();
  }, [selected]);

  // 針對非完成項目做輪詢 (每 5s)
  useEffect(() => {
    const notDone = items.filter((v) => v.status === "queued" || v.status === "processing");
    if (notDone.length === 0) return;
    let alive = true;
    const iv = setInterval(async () => {
      try {
        await Promise.all(
          notDone.map(async (v) => {
            const j = await pollVideo(v.id);
            if (!alive) return;
            setItems((prev) => prev.map((p) => (p.id === v.id ? { ...p, ...j } : p)));
          })
        );
      } catch (_) {}
    }, 5000);
    return () => { alive = false; clearInterval(iv); };
  }, [items, setItems]);

  const onDrop = async (files) => {
    if (!files?.length) return;
    const file = files[0];
    setUploading(true); setProgress(0);
    try {
      const created = await uploadFileXHR(file, { onProgress: (p) => setProgress(p.pct) });
      // 把新項目插到最前面
      setItems((prev) => [{ ...created }, ...prev]);
      setSelected(created);
    } catch (e) {
      alert(`上傳失敗: ${e.message}`);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const handleFilePick = (e) => onDrop(e.target.files);

  const handleDelete = async (v) => {
    if (!confirm(`確定刪除「${v.filename || v.title || v.id}」？`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/videos/${v.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
      setItems((prev) => prev.filter((x) => x.id !== v.id));
      if (selected?.id === v.id) { setSelected(null); setHighlights(null); }
    } catch (e) {
      alert(`刪除失敗: ${e.message}`);
    }
  };

  return (
    <div className="min-h-[100vh] bg-neutral-50 text-neutral-900">
      {/* Header */}
      <header className="sticky top-0 z-30 backdrop-blur bg-white/70 border-b border-neutral-200">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Film className="size-5" />
            <span className="font-semibold">五人制足球自動剪輯精華系統</span>
          </div>
          <div className="text-sm text-neutral-500">單頁工作室 · 上傳 / 查詢 / 精華播放&下載</div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 space-y-10">
        {/* 1) 上傳區 */}
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">上傳影片</h2>
            <div className="text-sm text-neutral-500">支援 MP4 / MOV · 單檔上傳</div>
          </div>

          <div
            className="rounded-2xl border border-dashed border-neutral-300 bg-white p-6 grid place-items-center text-center hover:border-neutral-400 transition"
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; }}
            onDrop={(e) => { e.preventDefault(); onDrop(e.dataTransfer.files); }}
          >
            <UploadCloud className="size-8 mb-2" />
            <p className="font-medium">拖曳影片到這裡，或</p>
            <div className="mt-3">
              <button
                onClick={() => inputRef.current?.click()}
                className="inline-flex items-center gap-2 rounded-xl bg-neutral-900 text-white px-4 py-2 text-sm hover:bg-neutral-800"
                disabled={uploading}
              >
                {uploading ? <Loader2 className="size-4 animate-spin"/> : <ChevronRight className="size-4"/>}
                選擇檔案
              </button>
              <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={handleFilePick} />
            </div>
            {uploading && (
              <div className="mt-4 w-full max-w-md text-left">
                <div className="mb-1 text-sm text-neutral-600">上傳中... {progress}%</div>
                <div className="h-2 w-full rounded-full bg-neutral-200 overflow-hidden">
                  <div className="h-full bg-neutral-900 transition-all" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}
            <div className="mt-3 text-xs text-neutral-500">檔案大小上限由後端限制 · 請保持此分頁開啟直到上傳完成</div>
          </div>
        </section>

        {/* 2) 歷史清單 */}
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">歷史影片</h2>
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-4 text-neutral-400" />
                <input
                  className="pl-8 pr-3 py-2 rounded-xl border bg-white text-sm w-64 outline-none focus:ring-2 focus:ring-neutral-300"
                  placeholder="搜尋檔名/ID..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
              <button className="text-sm rounded-xl border px-3 py-2 hover:bg-white" onClick={reload}>重新整理</button>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {loading && (
              <div className="col-span-full text-sm text-neutral-500">載入中...</div>
            )}
            {error && (
              <div className="col-span-full text-sm text-rose-600">載入失敗：{error}</div>
            )}
            {filtered.map((v) => (
              <motion.div key={v.id}
                layout
                className={`rounded-2xl border bg-white p-4 flex flex-col gap-3 ${selected?.id===v.id ? 'ring-2 ring-neutral-900' : ''}`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="font-medium truncate" title={v.filename || v.title || v.id}>
                    {v.filename || v.title || v.id}
                  </div>
                  <StatusBadge status={v.status} />
                </div>
                <div className="text-xs text-neutral-500 flex items-center gap-3">
                  <span>ID: {v.id}</span>
                  {v.durationSec != null && <span className="inline-flex items-center gap-1"><Clock className="size-3"/>{fmtDuration(v.durationSec)}</span>}
                  {v.sizeBytes != null && <span>{fmtBytes(v.sizeBytes)}</span>}
                  {v.createdAt && <span>{timeAgo(v.createdAt)}</span>}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="inline-flex items-center gap-2 rounded-xl px-3 py-2 border hover:bg-neutral-50 text-sm"
                    onClick={() => setSelected(v)}
                  >
                    <Play className="size-4"/> 檢視
                  </button>
                  {v.status === 'done' && v.archiveUrl && (
                    <a href={v.archiveUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-xl px-3 py-2 border hover:bg-neutral-50 text-sm">
                      <Download className="size-4"/> 全部精華下載
                    </a>
                  )}
                  <button
                    className="ml-auto inline-flex items-center gap-2 rounded-xl px-3 py-2 border hover:bg-rose-50 text-sm text-rose-700"
                    onClick={() => handleDelete(v)}
                  >
                    <Trash2 className="size-4"/> 刪除
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
          {filtered.length === 0 && !loading && (
            <div className="text-sm text-neutral-500 mt-4">沒有資料，先上傳一支影片吧。</div>
          )}
        </section>

        {/* 3) 精華區 (依背號分組) */}
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">精華剪輯</h2>
            <div className="text-sm text-neutral-500">點選上方某支影片後顯示</div>
          </div>

          {!selected && (
            <div className="rounded-2xl border bg-white p-6 text-neutral-500">尚未選擇影片。</div>
          )}

          {selected && selected.status !== 'done' && (
            <div className="rounded-2xl border bg-white p-6">
              <div className="flex items-center gap-2 text-neutral-700">
                <Loader2 className="size-4 animate-spin"/>
                <div>
                  影片「{selected.filename || selected.title || selected.id}」處理中...
                  <div className="text-sm text-neutral-500">完成後會自動載入精華（每 5 秒輪詢）</div>
                </div>
              </div>
            </div>
          )}

          {selected && selected.status === 'done' && (
            <div className="rounded-2xl border bg-white p-4">
              <div className="flex items-center justify-between gap-3 p-2">
                <div className="font-medium truncate">{selected.filename || selected.title || selected.id}</div>
                <div className="text-xs text-neutral-500 flex items-center gap-3">
                  {selected.durationSec != null && <span className="inline-flex items-center gap-1"><Clock className="size-3"/>{fmtDuration(selected.durationSec)}</span>}
                  {selected.sizeBytes != null && <span>{fmtBytes(selected.sizeBytes)}</span>}
                </div>
              </div>

              {hlLoading && <div className="p-4 text-sm text-neutral-600 flex items-center gap-2"><Loader2 className="size-4 animate-spin"/> 載入精華中...</div>}
              {hlError && <div className="p-4 text-sm text-rose-600 flex items-center gap-2"><AlertTriangle className="size-4"/> 載入失敗：{hlError}</div>}

              {highlights && Object.keys(highlights).length > 0 ? (
                <div className="mt-2">
                  {Object.entries(highlights).map(([jerseyNo, clips]) => (
                    <div key={jerseyNo} className="border-t first:border-t-0">
                      <div className="px-3 py-2 bg-neutral-50 flex items-center justify-between">
                        <div className="text-sm font-semibold">背號 {jerseyNo}</div>
                        <div className="text-xs text-neutral-500">{clips.length} 段</div>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-3">
                        {clips.map((c, idx) => (
                          <div key={idx} className="rounded-xl border overflow-hidden bg-white">
                            <div className="aspect-video bg-neutral-200">
                              <video
                                className="w-full h-full object-contain bg-black"
                                src={c.url}
                                poster={c.thumbUrl || undefined}
                                controls
                                preload="metadata"
                              />
                            </div>
                            <div className="p-3 text-sm flex items-center justify-between gap-3">
                              <div className="text-neutral-600">{fmtDuration(c.startSec)} - {fmtDuration(c.endSec)} ({fmtDuration((c.endSec||0)-(c.startSec||0))})</div>
                              <a href={c.downloadUrl || c.url} download className="inline-flex items-center gap-2 rounded-lg px-3 py-1.5 border hover:bg-neutral-50">
                                <Download className="size-4"/> 下載
                              </a>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                (!hlLoading && !hlError) && <div className="p-4 text-sm text-neutral-600">此影片沒有可顯示的精華。</div>
              )}

              {selected.archiveUrl && (
                <div className="p-3 border-t flex justify-end">
                  <a className="inline-flex items-center gap-2 rounded-xl px-3 py-2 border hover:bg-neutral-50 text-sm" href={selected.archiveUrl}>
                    <Download className="size-4"/> 下載全部精華 (ZIP)
                  </a>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="border-t bg-white">
        <div className="mx-auto max-w-6xl px-4 py-4 text-xs text-neutral-500 flex items-center justify-between">
          <span>© {new Date().getFullYear()} Futsal Highlights Studio</span>
          <span>單頁應用 · React + Tailwind</span>
        </div>
      </footer>
    </div>
  );
}
