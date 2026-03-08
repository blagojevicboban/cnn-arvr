import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  return {
    base: './', // Use relative paths for GitHub Pages to avoid repo name mismatch
    plugins: [react(), tailwindcss()],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    worker: {
      format: 'es',
    },
    optimizeDeps: {
      // ensure the dev server pre-bundles TensorFlow so that
      // worker imports can be resolved correctly
      include: ['@tensorflow/tfjs'],
    },
    build: {
      rollupOptions: {
        // do not mark tfjs as external; we want it bundled (or
        // split into a manual chunk) rather than leaving an
        // unresolved import that blows up the dev server
        output: {
          manualChunks: (id) => {
            if (id.includes('@tensorflow')) {
              return 'tensorflow';
            }
          },
        },
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
