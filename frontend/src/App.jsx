import { useEffect } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import AppShell from "./components/AppShell";
import Landing from "./pages/Landing";
import Dashboard from "./pages/Dashboard";
import Editor from "./pages/Editor";
import Documents from "./pages/Documents";
import Compare from "./pages/Compare";
import About from "./pages/About";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

function NotFound() {
  return (
    <div className="grid min-h-screen place-items-center px-6 text-center">
      <div>
        <h1 className="text-4xl font-bold">Page not found</h1>
        <p className="mt-2 text-ink-500">The page you are looking for does not exist.</p>
        <Link to="/dashboard" className="btn btn-primary mt-6">Go to dashboard</Link>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route element={<AppShell />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/editor" element={<Editor />} />
          <Route path="/editor/:id" element={<Editor />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/about" element={<About />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
