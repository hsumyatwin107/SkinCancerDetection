import { NavLink } from "react-router-dom";
import { Activity, Info, ScanLine, ShieldPlus } from "lucide-react";

const navItems = [
  { label: "Home", to: "/", icon: Activity },
  { label: "Detect", to: "/detect", icon: ScanLine },
  { label: "About", to: "/about", icon: Info },
];

function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-blue-100/80 bg-white/80 backdrop-blur-lg">
      <nav className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3.5 md:px-6">
        <NavLink to="/" className="group flex items-center gap-2 text-slate-800">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-sm transition group-hover:scale-105">
            <ShieldPlus className="h-4 w-4" />
          </span>
          <span className="text-sm font-extrabold tracking-tight md:text-base">
            Skin Cancer Detection
          </span>
        </NavLink>

        <div className="flex items-center gap-1 rounded-xl border border-slate-200/80 bg-slate-50/80 p-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
                  isActive
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-600 hover:bg-white hover:text-slate-900"
                }`
              }
            >
              <item.icon className="h-3.5 w-3.5" />
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  );
}

export default Navbar;
