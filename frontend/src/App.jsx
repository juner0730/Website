// src/App.jsx
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import RequireAuth from "./RequireAuth.jsx";
import OnePage from "./onepage.jsx";
import Login from "./login.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <OnePage />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
