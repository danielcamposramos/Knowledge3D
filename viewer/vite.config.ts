import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    fs: {
      allow: ['..']
    },
    proxy: {
      '/examples': {
        target: 'http://localhost:5173', // This is not used, but required by the type definition
        rewrite: (path) => path.replace(/^\/examples/, '/../examples')
      }
    }
  }
});
