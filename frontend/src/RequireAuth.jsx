// src/RequireAuth.jsx
import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "./auth/AuthProvider.jsx";

export default function RequireAuth({ children }) {
  const { user, booted } = useAuth();
  const location = useLocation();

  // 等 AuthProvider 第一次 /me 完成
  if (!booted) return null;

  // 沒登入 → 只在不是 "/" 時才導去 "/"，避免在 "/" 自己導自己造成 loop
  if (!user) {
    if (location.pathname !== "/") {
      return <Navigate to="/" state={{ from: location }} replace />;
    }
    // 在 "/" 就直接顯示登入頁（由路由決定），這裡不導
    return null;
  }

  // 已登入 → 放行
  return children;
}
