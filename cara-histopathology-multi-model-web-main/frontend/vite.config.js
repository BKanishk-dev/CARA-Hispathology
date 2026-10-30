import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api and /samples to the FastAPI backend.
// Override the target with:  VITE_API_TARGET=http://host:8000 npm run dev
const target = process.env.VITE_API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target, changeOrigin: true, timeout: 180000 },
      "/samples": { target, changeOrigin: true },
    },
  },
  build: {
    chunkSizeWarningLimit: 1200,
  },
});
