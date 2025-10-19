// src/App.js
import React, { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import Header from "./components/Header";

// Lazy-load pages for better initial performance
const LandingPage = lazy(() => import("./pages/LandingPage"));
const UploadPage = lazy(() => import("./pages/UploadPage"));
const SignIn = lazy(() => import("./pages/SignIn"));
const ResultsPage = lazy(() => import("./pages/ResultsPage")); // optional

// Simple NotFound component
function NotFound() {
  return (
    <div style={{ padding: 40, textAlign: "center" }}>
      <h2>Page not found</h2>
      <p>The page you requested doesn't exist.</p>
      <p>
        <a href="/">Return to home</a>
      </p>
    </div>
  );
}

/*
  HeaderController is a small component that decides whether to show
  the Header based on the current pathname. Because useLocation()
  must be used inside a Router, we render this component inside BrowserRouter.
*/
function HeaderController() {
  const location = useLocation();

  // Hide header for these routes (exact or prefix). Add more if needed.
  const hideOn = ["/signin"];
  const hideHeader = hideOn.some((p) => location.pathname === p || location.pathname.startsWith(p + "/"));

  return hideHeader ? null : <Header />;
}

export default function App() {
  // Optional base path (set REACT_APP_BASE_URL in your .env if needed)
  const basename = process.env.REACT_APP_BASE_URL || "/";

  return (
    <BrowserRouter basename={basename}>
      {/* HeaderController decides whether to render the Header for the current route */}
      <HeaderController />

      {/* Suspense fallback while lazy-loaded pages load */}
      <Suspense fallback={<div style={{ padding: 24 }}>Loading…</div>}>
        <Routes>
          <Route index element={<LandingPage />} />
          <Route path="/" element={<LandingPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/signin" element={<SignIn />} />

          {/* optional results page */}
          <Route path="/results" element={<ResultsPage />} />

          {/* redirect example */}
          <Route path="/home" element={<Navigate to="/" replace />} />

          {/* 404 */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
