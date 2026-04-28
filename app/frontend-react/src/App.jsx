import { Suspense } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Footer from "./components/Footer";
import Navbar from "./components/Navbar";
import PageLoader from "./components/PageLoader";
import AboutPage from "./pages/AboutPage";
import DetectPage from "./pages/DetectPage";
import HomePage from "./pages/HomePage";
import ResultPage from "./pages/ResultPage";

function App() {
  const location = useLocation();

  return (
    <main className="flex min-h-screen flex-col bg-[radial-gradient(circle_at_top,_#e0ecff_0%,_#f8fbff_45%,_#ffffff_100%)]">
      <Navbar />
      <Suspense fallback={<PageLoader />}>
        <div key={location.pathname} className="route-fade">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/detect" element={<DetectPage />} />
            <Route path="/result" element={<ResultPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Suspense>
      <Footer />
    </main>
  );
}

export default App;
