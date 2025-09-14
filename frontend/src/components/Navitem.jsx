import { Link } from "react-router-dom";
import clsx from "clsx";

export default function NavItem({ to, icon, label, active, collapsed }) {
  return (
    <Link
      to={to}
      className={clsx(
        "group flex items-center gap-3 px-3 py-2 rounded-xl text-sm border",
        active
          ? "bg-neutral-900 text-white border-neutral-900"
          : "bg-white hover:bg-neutral-50 border-neutral-200 text-neutral-700"
      )}
    >
      <span className={clsx("shrink-0", active && "text-white")}>{icon}</span>
      {!collapsed && <span className="truncate">{label}</span>}
    </Link>
  );
}
