import SampleGallery from "../components/SampleGallery";

export default function GalleryPage({ samples, onPickSample, loading }) {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Sample image gallery</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
          One image per tissue class (19 total); a few are deliberately blurred or faded so you can see the image-quality
          score lower the CARA reliability. Drop your own images into <code className="font-mono">data/samples</code> and
          they appear here automatically.
        </p>
      </div>
      <SampleGallery
        samples={samples}
        onPick={onPickSample}
        disabled={loading}
        columns="grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
      />
    </div>
  );
}
