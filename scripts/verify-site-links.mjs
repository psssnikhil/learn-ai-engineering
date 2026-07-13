#!/usr/bin/env node
/**
 * Smoke-test built site for broken internal links (GitHub Pages pretty URLs).
 * Run after: npm run build:docs
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.join(__dirname, '..', 'site');
const BASE = process.env.SITE_URL || 'https://psssnikhil.github.io/learn-ai-engineering';

const checks = [
  '/',
  '/start-here/',
  '/learn/',
  '/learn/study-plans/',
  '/faq/',
  '/topic-map/',
  '/getting-started/',
  '/projects/build-these/',
  '/agent-engineering/',
  '/foundations/module-00-genai-foundations-from-nlp-to-transformers/',
  '/build/module-09-rag-retrieval-augmented-generation/',
  '/production/module-10-llmops-production-systems/',
];

async function main() {
  const bad = [];
  for (const p of checks) {
    const url = `${BASE}${p}`;
    const res = await fetch(url, { method: 'HEAD', redirect: 'follow' });
    if (!res.ok) bad.push(`${res.status} ${url}`);
    else console.log(`OK ${url}`);
  }

  const indexHtml = fs.readFileSync(path.join(SITE, 'index.html'), 'utf-8');
  const mdHrefs = indexHtml.match(/href="[^"]*\.md"/g);
  if (mdHrefs?.some((h) => !h.includes('github.com'))) {
    bad.push(`index.html has broken .md hrefs: ${mdHrefs.join(', ')}`);
  }

  if (bad.length) {
    console.error('\nFAILED:');
    bad.forEach((b) => console.error(' ', b));
    process.exit(1);
  }
  console.log(`\nAll ${checks.length} URLs OK`);
}

main();
