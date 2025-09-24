import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "./auth/AuthProvider";

// 這份版型是「onepage testing.jsx」的視覺移植，但用純 inline CSS，不需要 Tailwind。
// 同時相容你既有的全域 API：
//   window.apiListVideos(token)
//   window.apiUpload(file, token)
//   window.apiListClips(videoId, token)
//   window.downloadFile(videoId, filename)

export default function OnePage() {
  const { user, logout, token } = useAuth();

  const [items, setItems] = useState([]);     // 影片列表
  const [selected, setSelected] = useState(null); // 被點選的影片
  const [clips, setClips] = useState([]);     // 精華清單
  const [query, setQuery] = useState("");     // 搜尋
  const [err, setErr] = useState("");         // 上傳錯誤
  const [uploading, setUploading] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const inputRef = useRef(null);

  // 載入影片列表
  const loadList = async () => {
    setErr("");
    try {
      const list = await window.apiListVideos(token);
      setItems(Array.isArray(list) ? list : []);
    } catch (e) {
      setErr(e?.message || String(e));
    }
  };
  useEffect(() => { loadList(); }, []);

  // 選擇影片後載入 clips
  useEffect(() => {
    if (!selected) { setClips([]); return; }
    window.apiListClips(selected.id, token)
      .then((arr) => setClips(Array.isArray(arr) ? arr : []))
      .catch(() => setClips([]));
  }, [selected, token]);

  // 內建一個有進度條的 XHR 上傳（fetch 沒上傳進度事件）
  const uploadWithProgress = (file) =>
    new Promise((resolve, reject) => {
      const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
      const xhr = new XMLHttpRequest();
      // 先嘗試 /api/upload，失敗再退回 /upload
      const tryOnce = (url) => {
        xhr.open("POST", url, true);
        xhr.withCredentials = true;
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) setProgressPct(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onreadystatechange = () => {
          if (xhr.readyState === 4) {
            if (xhr.status >= 200 && xhr.status < 300) {
              try { resolve(JSON.parse(xhr.responseText)); }
              catch (e) { resolve(null); }
            } else if (url.endsWith("/api/upload")) {
              tryOnce(`${API_BASE}/upload`);
            } else {
              reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
            }
          }
        };
        const fd = new FormData();
        fd.append("file", file);
        xhr.send(fd);
      };
      tryOnce(`${API_BASE}/api/upload`);
    });

  const onPick = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setErr(""); setUploading(true); setProgressPct(0);
    try {
      // 你的相容 API（若存在）優先使用；否則用內建 XHR
      if (typeof window.apiUpload === "function") {
        await window.apiUpload(f, token);
      } else {
        await uploadWithProgress(f);
      }
      await loadList();
      e.target.value = "";
    } catch (ex) {
      setErr("上傳失敗：" + (ex?.message || ex));
    } finally {
      setUploading(false);
      setProgressPct(0);
    }
  };

  const onDrop = async (files) => {
    const f = files?.[0];
    if (!f) return;
    setErr(""); setUploading(true); setProgressPct(0);
    try {
      if (typeof window.apiUpload === "function") {
        await window.apiUpload(f, token);
      } else {
        await uploadWithProgress(f);
      }
      await loadList();
    } catch (ex) {
      setErr("上傳失敗：" + (ex?.message || ex));
    } finally {
      setUploading(false);
      setProgressPct(0);
    }
  };

  const filtered = useMemo(() => {
    const key = query.trim().toLowerCase();
    if (!key) return items;
    return items.filter(v =>
      [v.original_name, v.id].filter(Boolean).some(s => String(s).toLowerCase().includes(key))
    );
  }, [items, query]);

  // ====== UI ======
  const S = styles; // 簡寫
  return (
    <div style={S.page}>
      {/* Header */}
      <header style={S.headerWrap}>
        <div style={S.headerInner}>
          <div style={S.brand}>
            <div style={S.brandDot} />
            <span style={S.brandText}>五人制足球AI運動員追蹤與精彩時刻捕捉系統</span>
          </div>
          <div style={S.headerRight}>
            {user?.email && <span style={S.headerEmail}>{user.email}</span>}
            <button onClick={logout} style={S.logoutBtn} title="登出">登出</button>
          </div>
        </div>
      </header>

      <main style={S.main}>
        {/* 上傳區 */}
        <section>
          <div style={S.sectionHead}>
            <h2 style={S.sectionTitle}>上傳影片</h2>
            <div style={S.sectionNote}>支援 MP4 / MOV · 單檔上傳</div>
          </div>

          <div
            style={S.uploadCard}
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; }}
            onDrop={(e) => { e.preventDefault(); onDrop(e.dataTransfer.files); }}
          >
            <div style={{ fontSize: 38, marginBottom: 8 }}>📤</div>
            <div style={{ fontWeight: 600 }}>拖曳影片到這裡，或</div>

            <div style={{ marginTop: 12 }}>
              <button
                onClick={() => inputRef.current?.click()}
                disabled={!!uploading}
                style={{ ...S.primaryBtn, opacity: uploading ? 0.7 : 1 }}
              >
                {uploading ? "上傳中…" : "選擇檔案"}
              </button>
              <input
                ref={inputRef}
                type="file"
                accept="video/*"
                onChange={onPick}
                style={{ display: "none" }}
                disabled={!!uploading}
              />
            </div>

            {uploading && (
              <div style={{ marginTop: 16, width: "100%", maxWidth: 520, textAlign: "left" }}>
                <div style={{ marginBottom: 6, fontSize: 13, color: "#525252" }}>上傳中… {progressPct}%</div>
                <div style={S.progressOuter}>
                  <div style={{ ...S.progressInner, width: `${progressPct}%` }} />
                </div>
              </div>
            )}

            <div style={S.uploadHint}>檔案大小上限由後端限制 · 請保持此分頁開啟直到上傳完成</div>
            {!!err && <div style={S.errText}>{err}</div>}
          </div>
        </section>

        {/* 歷史清單 */}
        <section>
          <div style={S.sectionHead}>
            <h2 style={S.sectionTitle}>歷史影片</h2>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input
                placeholder="搜尋檔名/ID..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={S.searchInput}
              />
              <button onClick={loadList} style={S.secondaryBtn}>重新整理</button>
            </div>
          </div>

          <div style={S.cardsGrid}>
            {filtered.map((v) => (
              <div
                key={v.id}
                style={{
                  ...S.card,
                  ...(selected?.id === v.id ? S.cardActive : null),
                }}
                onClick={() => setSelected(v)}
                title={v.original_name || v.id}
              >
                <div style={S.cardTop}>
                  <div style={S.cardTitle} title={v.original_name || v.id}>
                    {v.original_name || v.id}
                  </div>
                  <span style={{ ...S.badge, ...badgeStyle(v.status) }}>{v.status}</span>
                </div>
                <div style={S.cardMeta}>
                  {v.created_at ? new Date(v.created_at).toLocaleString() : "-"}
                </div>
              </div>
            ))}
          </div>

          {!filtered.length && (
            <div style={{ marginTop: 6, fontSize: 13, color: "#6b7280" }}>沒有資料，先上傳一支影片吧。</div>
          )}
        </section>

        {/* 精華剪輯 */}
        <section>
          <div style={S.sectionHead}>
            <h2 style={S.sectionTitle}>精華剪輯</h2>
            <div style={S.sectionNote}>點上方某支影片後顯示</div>
          </div>

          {!selected && (
            <div style={S.clipCard}>尚未選擇影片。</div>
          )}

          {selected && (
            <div style={S.clipWrap}>
              <div style={S.clipHeader}>
                <div style={{ ...S.cardTitle, maxWidth: "70%" }}>{selected.original_name || selected.id}</div>
                <div style={{ fontSize: 12, color: "#6b7280" }}>{selected.status}</div>
              </div>

              {clips && clips.length ? (
                <div style={S.clipGrid}>
                  {clips.map((c) => (
                    <div key={c.filename} style={S.clipItem}>
                      <div style={S.videoBox}>
                        <video
                          style={S.video}
                          src={`/files/${encodeURIComponent(selected.id)}/${encodeURIComponent(c.filename)}`}
                          controls
                          preload="metadata"
                        />
                      </div>
                      <div style={S.clipFooter}>
                        <div style={S.clipName} title={c.filename}>{c.filename}</div>
                        <button
                          onClick={() => window.downloadFile(selected.id, c.filename)}
                          style={S.secondaryBtn}
                        >
                          下載
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ padding: 14, fontSize: 14, color: "#525252" }}>
                  尚未產生精華剪輯（以球員背號分類）。
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer style={S.footer}>
        <div style={S.footerInner}>
          <span>© {new Date().getFullYear()} Futsal Highlights Studio</span>
          <span>單頁應用 · React</span>
        </div>
      </footer>
    </div>
  );
}

/* ===== 風格（純 inline CSS） ===== */

const styles = {
  page: { minHeight: "100vh", background: "#f8fafc", color: "#111827" },

  headerWrap: {
    position: "sticky", top: 0, zIndex: 30, backdropFilter: "blur(6px)",
    background: "rgba(255,255,255,.7)", borderBottom: "1px solid #e5e7eb",
  },
  headerInner: {
    maxWidth: 1200, margin: "0 auto", padding: "12px 16px",
    display: "flex", alignItems: "center", justifyContent: "space-between",
  },
  brand: { display: "flex", alignItems: "center", gap: 8 },
  brandDot: { width: 10, height: 10, borderRadius: 999, background: "#111827" },
  brandText: { fontWeight: 600 },
  headerRight: { display: "flex", alignItems: "center", gap: 10 },
  headerEmail: { fontSize: 13, color: "#6b7280" },
  logoutBtn: {
    fontSize: 13, borderRadius: 12, border: "1px solid #e5e7eb",
    padding: "8px 12px", background: "white", cursor: "pointer",
  },

  main: { maxWidth: 1200, margin: "0 auto", padding: "24px 16px", display: "grid", gap: 40 },

  sectionHead: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 },
  sectionTitle: { margin: 0, fontSize: 18, fontWeight: 600 },
  sectionNote: { fontSize: 13, color: "#6b7280" },

  uploadCard: {
    borderRadius: 16, border: "1px dashed #d1d5db", background: "#ffffff",
    padding: 24, textAlign: "center",
  },
  uploadHint: { marginTop: 10, fontSize: 12, color: "#6b7280" },
  errText: { marginTop: 8, fontSize: 14, color: "#dc2626" },

  primaryBtn: {
    borderRadius: 12, padding: "8px 14px", fontSize: 14, fontWeight: 600,
    background: "#111827", color: "#fff", border: "1px solid #111827", cursor: "pointer",
  },
  secondaryBtn: {
    borderRadius: 10, padding: "8px 12px", fontSize: 13,
    background: "white", color: "#111827", border: "1px solid #e5e7eb", cursor: "pointer",
  },

  progressOuter: { height: 8, width: "100%", borderRadius: 999, background: "#e5e7eb", overflow: "hidden" },
  progressInner: { height: "100%", background: "#111827", transition: "width .15s linear" },

  searchInput: {
    width: 260, padding: "8px 12px", borderRadius: 12, border: "1px solid #e5e7eb",
    background: "#fff", fontSize: 14, outline: "none",
  },

  cardsGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(260px,1fr))", gap: 12 },
  card: { borderRadius: 16, border: "1px solid #e5e7eb", background: "#fff", padding: 14, cursor: "pointer" },
  cardActive: { outline: "2px solid #111827" },
  cardTop: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 },
  cardTitle: { fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },
  cardMeta: { fontSize: 12, color: "#6b7280" },

  clipCard: { borderRadius: 16, border: "1px solid #e5e7eb", background: "#fff", padding: 24, color: "#6b7280" },
  clipWrap: { borderRadius: 16, border: "1px solid #e5e7eb", background: "#fff", padding: 12 },
  clipHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: 8 },
  clipGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(320px,1fr))", gap: 12 },
  clipItem: { borderRadius: 12, border: "1px solid #e5e7eb", background: "#fff", overflow: "hidden" },
  videoBox: { aspectRatio: "16/9", background: "#000" },
  video: { width: "100%", height: "100%", objectFit: "contain", background: "#000" },
  clipFooter: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: 10 },
  clipName: { fontSize: 13, color: "#374151", maxWidth: "70%", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },

  footer: { borderTop: "1px solid #e5e7eb", background: "#fff", marginTop: 24 },
  footerInner: {
    maxWidth: 1200, margin: "0 auto", padding: "12px 16px",
    fontSize: 12, color: "#6b7280", display: "flex", alignItems: "center", justifyContent: "space-between",
  },
};

// 狀態徽章色
function badgeStyle(status) {
  switch (status) {
    case "processing":
      return { background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe", padding: "2px 8px", borderRadius: 999, fontSize: 12 };
    case "done":
      return { background: "#ecfdf5", color: "#047857", border: "1px solid #a7f3d0", padding: "2px 8px", borderRadius: 999, fontSize: 12 };
    case "failed":
      return { background: "#fef2f2", color: "#b91c1c", border: "1px solid #fecaca", padding: "2px 8px", borderRadius: 999, fontSize: 12 };
    case "queued":
      return { background: "#fffbeb", color: "#b45309", border: "1px solid #fde68a", padding: "2px 8px", borderRadius: 999, fontSize: 12 };
    default:
      return { background: "#f5f5f5", color: "#374151", border: "1px solid #e5e7eb", padding: "2px 8px", borderRadius: 999, fontSize: 12 };
  }
}
