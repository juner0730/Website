import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const AuthCtx = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem("app:user");
    return raw ? JSON.parse(raw) : null;
  });

  useEffect(() => {
    if (user) localStorage.setItem("app:user", JSON.stringify(user));
    else localStorage.removeItem("app:user");
  }, [user]);

  const login = (profile) => setUser(profile);
  const logout = () => setUser(null);

  const value = useMemo(() => ({ user, login, logout }), [user]);
  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
