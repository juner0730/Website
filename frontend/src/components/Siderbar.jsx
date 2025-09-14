import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Film, Download, Shirt, LogOut, Menu } from "lucide-react";
import NavItem from "./NavItem.jsx";
import { useAuth } from "../auth/AuthProvider.jsx";
import clsx from "clsx";

export default function Sidebar() {
  const [open, setOpen] = useState(true);
  const { pathname } = useLocation();
  const { logout, user } = useAuth();

  return (
    <>
      {/* Desktop sidebar */}
      <aside
        className={clsx(
          "hidden md:flex sticky top-0 h-screen shrink-0 border-r border-neutral-200 bg-white/70 backdrop-blur-sm",
          open ? "w-64" : "w-20"
        )}
      >
        <div className="flex flex-col w-full">
          <div className="h-16 flex items-center justify-between px-4">
            <Link to="/processed" className="font-semibold tracking-wide">
              {open ? "自動剪輯系統" : "剪輯"}
            </Link>
            <button
              onClick={() => setOpen((v) => !v)}
              className="p-2 rounded-xl hover:bg-neutral-100"
              aria-label="Toggle sidebar"
            >
              <Menu size={18} />
            </button>
          </div>

          <nav className="px-2 space-y-1">
            <NavItem
              to="/processed"
              icon={<Film size={18} />}
              active={pathname.startsWith("/processed")}
              label="影片處理好的"
              collapsed={!open}
            />
            <NavItem
              to="/downloads"
              icon={<Download size={18} />}
              active={pathname.startsWith("/downloads")}
              label="影片下載"
              collapsed={!open}
            />
            <NavItem
              to="/jerseys"
              icon={<Shirt size={18} />}
              active={pathname.startsWith("/jerseys")}
              label="背號"
              collapsed={!open}
            />
          </nav>

          <div className="mt-auto p-3">
            <button
              onClick={logout}
              className={clsx(
                "w-full flex items-center gap-3 px-3 py-2 rounded-xl border text-sm",
                "hover:bg-neutral-50 active:bg-neutral-100 border-neutral-200"
              )}
            >
              <LogOut size={18} />
              {open && <span>登出</span>}
            </button>
            {open && user && (
              <p className="text-xs text-neutral-500 mt-3 px-1">
                已登入：{user.name}
              </p>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile top button to open nav (可依需求擴充為抽屜) */}
      <div className="md:hidden fixed left-4 top-4 z-40">
        <Link
          to="/processed"
          className="px-3 py-2 rounded-xl bg-white/80 backdrop-blur shadow-sm border border-neutral-200 text-sm"
        >
          菜單
        </Link>
      </div>
    </>
  );
}
