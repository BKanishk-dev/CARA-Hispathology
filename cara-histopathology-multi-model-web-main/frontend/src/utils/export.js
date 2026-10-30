import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const csvCell = (v) => {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

const toCsv = (rows) => rows.map((r) => r.map(csvCell).join(",")).join("\n");
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
const baseName = (name) => (name || "image").replace(/\.[^.]+$/, "");

const ANALYSIS_HEADER = [
  "rank", "model", "origin", "predicted_class", "softmax_confidence", "image_quality",
  "class_reliability", "cara_score", "class_threshold", "decision", "inference_ms",
];

function analysisRows(result) {
  const byRank = [...result.models].sort((a, b) => a.rank - b.rank);
  return byRank.map((m) => [
    m.rank, m.name, m.origin, m.predicted_label, m.confidence.toFixed(4), m.quality.toFixed(4),
    m.class_reliability.toFixed(4), m.cara_score.toFixed(4), m.threshold.toFixed(4), m.decision,
    m.inference_ms == null ? "" : m.inference_ms.toFixed(2),
  ]);
}

export function exportAnalysisCSV(result, fileName) {
  const lines = [];
  
  lines.push(`# image: ${fileName || "uploaded image"}`, `# mode: ${result.mode}`);
  const csv = lines.join("\n") + "\n" + toCsv([ANALYSIS_HEADER, ...analysisRows(result)]);
  download(new Blob([csv], { type: "text/csv;charset=utf-8" }), `cara_${baseName(fileName)}_${stamp()}.csv`);
}

function toJpegDataUrl(src, maxSide = 360) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      resolve({ url: c.toDataURL("image/jpeg", 0.88), w: c.width, h: c.height });
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

export async function exportAnalysisPDF(result, fileName, previewUrl) {
  const doc = new jsPDF({ orientation: "landscape", unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  let y = 40;

  doc.setFont("helvetica", "bold").setFontSize(18).text("CARA - Multi-model histopathology report", 40, y);
  y += 20;
  doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(90);
  doc.text(`Image: ${fileName || "uploaded image"}   |   Generated: ${new Date().toLocaleString()}   |   Mode: ${result.mode}`, 40, y);
  y += 16;
  
  doc.setTextColor(0);

  const thumb = previewUrl ? await toJpegDataUrl(previewUrl) : null;
  let tableStartX = 40;
  if (thumb) {
    const h = 150;
    const w = (thumb.w / thumb.h) * h;
    doc.addImage(thumb.url, "JPEG", 40, y, w, h);
    doc.setFontSize(9).setTextColor(90);
    doc.text(`Image quality Q = ${result.image.quality.toFixed(3)}`, 40 + w + 14, y + 14);
    doc.text(`Sharpness = ${result.image.blur.toFixed(3)}   Staining = ${result.image.stain.toFixed(3)}`, 40 + w + 14, y + 28);
    doc.text(`Consensus: ${result.consensus.label} (${result.consensus.votes}/${result.consensus.total} models)`, 40 + w + 14, y + 42);
    doc.setTextColor(0);
    y += h + 16;
  }

  autoTable(doc, {
    startY: y,
    margin: { left: tableStartX, right: 40 },
    head: [["#", "Model", "Predicted class", "Confidence", "Quality", "Class reliability", "CARA score", "Threshold", "Decision", "Time (ms)"]],
    body: [...result.models].sort((a, b) => a.rank - b.rank).map((m) => [
      m.rank, m.name, m.predicted_label, `${(m.confidence * 100).toFixed(1)}%`, m.quality.toFixed(3),
      m.class_reliability.toFixed(3), m.cara_score.toFixed(3), m.threshold.toFixed(3), m.decision,
      m.inference_ms == null ? "-" : m.inference_ms.toFixed(1),
    ]),
    styles: { fontSize: 9, cellPadding: 5 },
    headStyles: { fillColor: [79, 70, 229] },
    didParseCell: (d) => {
      if (d.section === "body" && d.column.index === 8) {
        d.cell.styles.textColor = d.cell.raw === "Reliable" ? [5, 150, 105] : [190, 18, 60];
        d.cell.styles.fontStyle = "bold";
      }
    },
  });

  const endY = doc.lastAutoTable.finalY + 18;
  doc.setFontSize(8).setTextColor(110);
  doc.text(
    "Reliable = CARA score R(x) = a*C + b*Q + g*S is at or above the class-specific threshold fitted on the validation set. " +
      "Unreliable predictions should be reviewed by an expert. Research prototype - not for clinical use.",
    40, endY, { maxWidth: W - 80 },
  );
  doc.save(`cara_${baseName(fileName)}_${stamp()}.pdf`);
}

export function exportDashboardCSV(results) {
  const header = [
    "model", "origin", "accuracy", "precision", "recall", "f1", "coverage", "rejection_rate",
    "selective_accuracy", "params_millions", "latency_ms",
  ];
  const rows = results.ranking.map((k) => {
    const m = results.models[k];
    const x = m.metrics;
    return [m.display_name, m.origin, x.accuracy, x.precision, x.recall, x.f1, x.coverage, x.rejection_rate,
      x.selective_accuracy, m.params_millions, m.latency_ms];
  });
  
  download(new Blob([toCsv([header, ...rows])], { type: "text/csv;charset=utf-8" }), `cara_dashboard_${stamp()}.csv`);
}
