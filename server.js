import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const HOST = '0.0.0.0';
const WEB_DIR = path.join(__dirname, 'web');

// Serve the static PlayQ UI and pre-rendered demo media (supports HEAD and Range requests)
app.use(express.static(WEB_DIR, {
  index: 'index.html',
  fallthrough: true,
}));

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'playq-web' });
});

// Ensure missing demo or asset files return 404 so HEAD checks in web/app.js work accurately
app.use(['/demo', '/assets'], (_req, res) => {
  res.status(404).end();
});

// Fallback to index.html for top-level page requests without a file extension
app.use((req, res) => {
  if (req.method !== 'GET' || path.extname(req.path)) {
    return res.status(404).end();
  }
  return res.sendFile(path.join(WEB_DIR, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`PlayQ web server listening on http://${HOST}:${PORT}`);
});
