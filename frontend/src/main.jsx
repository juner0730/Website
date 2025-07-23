import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import App from "./App.jsx";
import Results from "./results.jsx";
import PlayerResults from "./playerresults.jsx";
import Login from "./login.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/upload" element={<App />} />
        <Route path="/results" element={<Results />} />
        <Route path="/playerresults/:id" element={<PlayerResults />} />
      </Routes>
    </Router>
  </React.StrictMode>
);
