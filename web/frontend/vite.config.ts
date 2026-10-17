import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: frontend ở 5173, gọi API / WebSocket của FastAPI ở 8000 qua proxy
export default defineConfig({
  plugins: [react()],
  base: "./", // đường dẫn tương đối: chạy được cả trên hosting tĩnh lẫn sau FastAPI
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
