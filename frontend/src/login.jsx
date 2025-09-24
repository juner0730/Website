import React, { useState } from "react";
const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function Login() {
  const [busy, setBusy] = useState(false);
  return (
    <div style={{
      minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center",
      background:"#0b1020"
    }}>
      <div style={{
        width: 360, padding: 28, borderRadius: 16,
        background: "rgba(255,255,255,.06)", backdropFilter:"blur(6px)",
        border: "1px solid rgba(255,255,255,.12)", color:"#fff", boxShadow:"0 10px 30px rgba(0,0,0,.4)"
      }}>
        <h1 style={{margin:0, marginBottom: 16, fontSize:20, fontWeight:700, letterSpacing:.5, textAlign:"center"}}>
          五人制足球AI運動員追蹤與精彩時刻捕捉系統
        </h1>
        <p style={{opacity:.8, fontSize:13, textAlign:"center", marginTop:0, marginBottom:18}}>
          歡迎使用此系統
        </p>
        <button
          onClick={() => {
            setBusy(true);
            const next = `${window.location.origin}/onepage`;
            window.location.href = `${API}/auth/google/start?next=${encodeURIComponent(next)}`;
          }}
          disabled={busy}
          style={{
            width:"100%", padding:"12px 14px", borderRadius:12, border:"1px solid rgba(255,255,255,.2)",
            background: busy ? "#1456ad" : "#1a73e8", color:"#fff", fontWeight:700, cursor: busy ? "not-allowed":"pointer"
          }}
        >{busy ? "前往 Google…" : "使用 Google 登入"}</button>
      </div>
    </div>
  );
}
