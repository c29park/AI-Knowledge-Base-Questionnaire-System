import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    open: true, // opens http://localhost:5173 in your browser
    proxy: {"/api": "http://localhost:8000"}
  },
});