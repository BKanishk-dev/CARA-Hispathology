// Thin fetch wrapper around the FastAPI backend.
// In dev, Vite proxies /api and /samples to http://localhost:8000. For a separately hosted
// backend set VITE_API_URL (e.g. https://cara.example.com) at build time.
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch (err) {
    throw new Error("Cannot reach the API. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  models: () => request("/api/models"),
  classes: () => request("/api/classes"),
  results: () => request("/api/results"),
  samples: () => request("/api/samples"),
  compare(file, referenceClass) {
    const form = new FormData();
    form.append("file", file);
    if (referenceClass) form.append("reference_class", referenceClass);
    return request("/api/compare", { method: "POST", body: form });
  },
  explain(modelKey, file, targetClass) {
    const form = new FormData();
    form.append("file", file);
    if (targetClass != null) form.append("target_class", targetClass);
    return request(`/api/explain/${modelKey}`, { method: "POST", body: form });
  },
};

export const assetUrl = (path) => `${BASE}${path}`;

/** Download a gallery sample and wrap it in a File so it can go through the normal upload path. */
export async function fetchSampleAsFile(sample) {
  const res = await fetch(assetUrl(sample.url));
  if (!res.ok) throw new Error("Could not load the sample image");
  const blob = await res.blob();
  return new File([blob], sample.file, { type: blob.type || "image/jpeg" });
}
