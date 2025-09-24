import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, { credentials: "include", ...opts });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : null;
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [booted, setBooted] = useState(false);

  useEffect(() => {
    let mounted = true;
    fetchJSON(`${API_BASE}/me`)
      .then((u) => mounted && setUser(u))
      .catch(() => mounted && setUser(null))
      .finally(() => (mounted && setBooted(true)));
    return () => (mounted = false);
  }, []);

  const login = () => {
    const next = `${window.location.origin}/onepage`;
    window.location.href = `${API_BASE}/auth/google/start?next=${encodeURIComponent(next)}`;
  };

  const logout = async () => {
    try { await fetchJSON(`${API_BASE}/logout`, { method: "POST" }); } catch {}
    setUser(null);
  };

  const value = useMemo(() => ({ user, booted, login, logout }), [user, booted]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

export default AuthProvider;
