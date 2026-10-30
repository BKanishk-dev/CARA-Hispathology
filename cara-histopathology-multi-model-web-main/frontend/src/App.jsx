import { useCallback, useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import ModeBanner from "./components/ModeBanner";
import AnalyzePage from "./pages/AnalyzePage";
import DashboardPage from "./pages/DashboardPage";
import GalleryPage from "./pages/GalleryPage";
import { api, fetchSampleAsFile } from "./api/client";
import useTheme from "./hooks/useTheme";

const EMPTY = { file: null, previewUrl: null, referenceClass: null, result: null, loading: false, error: null };

export default function App() {
  const { theme, toggle } = useTheme();
  const [page, setPage] = useState("analyze");
  const [health, setHealth] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [results, setResults] = useState(null);
  const [resultsError, setResultsError] = useState(null);
  const [analysis, setAnalysis] = useState(EMPTY);
  const urlRef = useRef(null);

  // initial data (the backend may still be starting, so retry the health check a few times)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let attempt = 0; attempt < 30 && !cancelled; attempt++) {
        try {
          const h = await api.health();
          if (!cancelled) {
            setHealth(h);
            setApiError(null);
          }
          break;
        } catch (e) {
          if (!cancelled) setApiError(attempt < 2 ? null : `${e.message} (retrying…)`);
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
      if (cancelled) return;
      api.samples().then((s) => !cancelled && setSamples(s)).catch(() => {});
      api.results().then((r) => !cancelled && setResults(r)).catch((e) => !cancelled && setResultsError(e.message));
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const setFile = useCallback((file, referenceClass = null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const previewUrl = URL.createObjectURL(file);
    urlRef.current = previewUrl;
    setAnalysis({ ...EMPTY, file, previewUrl, referenceClass });
  }, []);

  const clear = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setAnalysis(EMPTY);
  }, []);

  const runCompare = useCallback(async (file, referenceClass) => {
    setAnalysis((a) => ({ ...a, loading: true, error: null, result: null }));
    try {
      const result = await api.compare(file, referenceClass);
      setAnalysis((a) => ({ ...a, loading: false, result }));
    } catch (e) {
      setAnalysis((a) => ({ ...a, loading: false, error: e.message }));
    }
  }, []);

  const pickSample = useCallback(
    async (sample) => {
      setPage("analyze");
      window.scrollTo({ top: 0, behavior: "smooth" });
      try {
        const file = await fetchSampleAsFile(sample);
        setFile(file, sample.class);
        runCompare(file, sample.class);
      } catch (e) {
        setAnalysis((a) => ({ ...a, error: e.message }));
      }
    },
    [setFile, runCompare],
  );

  return (
    <div className="min-h-screen">
      <Header page={page} onNavigate={setPage} theme={theme} onToggleTheme={toggle} />
      <ModeBanner health={health} error={apiError} />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        {page === "analyze" && (
          <AnalyzePage
            state={analysis}
            samples={samples}
            onFile={(f) => setFile(f)}
            onClear={clear}
            onCompare={() => runCompare(analysis.file, analysis.referenceClass)}
            onPickSample={pickSample}
          />
        )}
        {page === "dashboard" && <DashboardPage results={results} error={resultsError} />}
        {page === "gallery" && <GalleryPage samples={samples} onPickSample={pickSample} loading={analysis.loading} />}
      </main>
    </div>
  );
}
