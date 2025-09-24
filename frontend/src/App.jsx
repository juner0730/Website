// src/App.jsx
import React from "react";
import { Routes, Route } from "react-router-dom";
import LoginPage from "./login.jsx";         // 你剛給的漂亮登入 UI
import OnePage from "./onepage.jsx";               // 你的 onepage，保持不動
import RequireAuth from "./RequireAuth.jsx";

export default function App() {
  return (
    <Routes>
      {/* 登入頁不加任何保護 */}
      <Route path="/" element={<LoginPage />} />

      {/* 只有 onepage 需要登入 */}
      <Route
        path="/onepage"
        element={
          <RequireAuth>
            <OnePage />
          </RequireAuth>
        }
      />
    </Routes>
  );
}
