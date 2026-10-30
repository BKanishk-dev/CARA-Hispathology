import { useState } from "react";
import { FileDown, FileSpreadsheet } from "lucide-react";
import { exportAnalysisCSV, exportAnalysisPDF } from "../utils/export";

export default function ExportMenu({ result, fileName, previewUrl }) {
  const [busy, setBusy] = useState(false);

  const pdf = async () => {
    setBusy(true);
    try {
      await exportAnalysisPDF(result, fileName, previewUrl);
    } catch (e) {
      alert(`PDF export failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap gap-2">
      <button className="btn-secondary" onClick={() => exportAnalysisCSV(result, fileName)}>
        <FileSpreadsheet size={16} /> Export CSV
      </button>
      <button className="btn-secondary" onClick={pdf} disabled={busy}>
        <FileDown size={16} /> {busy ? "Preparing…" : "Export PDF"}
      </button>
    </div>
  );
}
