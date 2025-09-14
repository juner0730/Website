import React from "react";
import { Link } from "react-router-dom";
export default function NotFound() {
  return (
    <div className="h-[60vh] grid place-items-center">
      <div className="text-center">
        <h1 className="text-2xl font-semibold mb-2">404 - Page Not Found</h1>
        <p className="text-neutral-600 mb-4">The page you are looking for does not exist.</p>
        <Link to="/results" className="underline">Go back</Link>
      </div>
    </div>
  );
}
