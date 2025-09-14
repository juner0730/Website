import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "./auth/AuthProvider.jsx";

export default function RequireAuth({ children }) {
  const { user } = useAuth();
  const loc = useLocation();
  if (!user) {
    return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  }
  return children;
}
