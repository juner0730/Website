import React from "react";
import { Outlet, Link } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider.jsx";

export default function Layout() {
  const { user, logout } = useAuth();

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <Link to="/upload" className="font-semibold">⚽ 五人制足球自動剪輯精華系統</Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link to="/upload">Upload</Link>
            <Link to="/results">Results</Link>
            {user ? (
              <div className="flex items-center gap-3">
                {user.picture && (
                  <img src={user.picture} alt="avatar" className="w-6 h-6 rounded-full" />
                )}
                <span>{user.name?.split(" ")[0] || "User"}</span>
                <button className="underline" onClick={logout}>Logout</button>
              </div>
            ) : (
              <Link to="/login" className="underline">Login</Link>
            )}
          </nav>
        </div>
      </header>
      <main className="px-4 py-6">
        <div className="mx-auto max-w-6xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
