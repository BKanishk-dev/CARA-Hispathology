import { Layers, Sparkles } from "lucide-react";
import Dropzone from "../components/Dropzone";
import ModelCard from "../components/ModelCard";
import RankingTable from "../components/RankingTable";
import ComparisonChart from "../components/ComparisonChart";
import ExportMenu from "../components/ExportMenu";
import SampleGallery from "../components/SampleGallery";
import Spinner from "../components/Spinner";
import StatTile from "../components/StatTile";
import { pct } from "../utils/format";
import { explainConsensus, explainImage } from "../utils/explain";

export default function AnalyzePage({ state, samples, onFile, onClear, onCompare, onPickSample }) {
  const { file, previewUrl, result, loading, error } = state;
  const orderedModels = result ? [...result.models].sort((a, b) => a.rank - b.rank) : [];
  const reliableCount = result ? result.models.filter((m) => m.reliable).length : 0;

  return (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-5">
        <div className="space-y-4 lg:col-span-3">
          <Dropzone file={file} previewUrl={previewUrl} onFile={onFile} onClear={onClear} disabled={loading} />
          <div className="flex flex-wrap items-center gap-3">
            <button className="btn-primary !px-5 !py-2.5" onClick={onCompare} disabled={!file || loading}>
              <Layers size={18} /> {loading ? "Analyzing…" : "Compare All Models"}
            </button>
            {result && !loading && <ExportMenu result={result} fileName={file?.name} previewUrl={previewUrl} />}
          </div>
          {error && (
            <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>

        <div className="card p-5 lg:col-span-2">
          <h2 className="flex items-center gap-2 text-sm font-bold">
            <Sparkles size={16} className="text-brand-600 dark:text-brand-300" /> How CARA decides
          </h2>
          <ol className="mt-3 space-y-2 text-sm text-slate-600 dark:text-slate-300">
            <li><strong>C</strong> – softmax confidence of the predicted class</li>
            <li><strong>Q</strong> – image quality (sharpness + H&amp;E staining)</li>
            <li><strong>S</strong> – class reliability measured on the validation set</li>
            <li>
              <strong>R = α·C + β·Q + γ·S</strong> is compared with a class-specific threshold that is
              calibrated on the validation set only.
            </li>
            <li>
              <span className="font-semibold text-emerald-700 dark:text-emerald-300">Reliable</span> if R ≥ threshold,
              otherwise <span className="font-semibold text-rose-700 dark:text-rose-300">Unreliable</span> → flag for
              expert review.
            </li>
          </ol>
        </div>
      </section>

      {loading && <Spinner label="Running 8 models and computing CARA scores…" />}

      {result && !loading && (
        <section className="space-y-6" aria-live="polite">
          

          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatTile label="Image quality Q" value={pct(result.image.quality)} sub="blur + staining" />
            <StatTile label="Sharpness" value={pct(result.image.blur)} sub="Laplacian variance" />
            <StatTile label="Staining" value={pct(result.image.stain)} sub="H&E presence, intensity, contrast" />
            <StatTile
              label="Consensus"
              value={result.consensus.label}
              sub={`${result.consensus.votes}/${result.consensus.total} models agree · ${reliableCount} reliable`}
            />
          </div>

          <div className="card space-y-2 p-5 text-sm text-slate-700 dark:text-slate-200">
            <h2 className="text-base font-bold">Summary in words</h2>
            <p><strong>Your image:</strong> {explainImage(result)}</p>
            <p><strong>Models:</strong> {explainConsensus(result)}</p>
          </div>

          <div>
            <h2 className="mb-3 text-lg font-bold">Side-by-side results</h2>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {orderedModels.map((m) => (
                <ModelCard key={m.key} m={m} consensusLabel={result.consensus.label} previewUrl={previewUrl} />
              ))}
            </div>
          </div>

          <div>
            <h2 className="mb-3 text-lg font-bold">Model ranking</h2>
            <RankingTable models={result.models} />
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
              Ranked by decision (reliable first), then CARA score. Click a column header to sort.
            </p>
          </div>

          <ComparisonChart models={orderedModels} />
        </section>
      )}

      <section>
        <h2 className="mb-1 text-lg font-bold">Sample images</h2>
        <p className="mb-3 text-sm text-slate-500 dark:text-slate-400">Click a sample to analyze it with all models.</p>
        <SampleGallery samples={samples.slice(0, 8)} onPick={onPickSample} disabled={loading} />
      </section>
    </div>
  );
}
