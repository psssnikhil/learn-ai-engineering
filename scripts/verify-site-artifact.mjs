#!/usr/bin/env node
/**
 * Verify built site artifact — no internal .md hrefs, key pages exist.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.join(__dirname, '..', 'site');

const required = [
  'index.html',
  ['start-here/index.html', 'start-here.html'],
  'learn/index.html',
  ['faq/index.html', 'faq.html'],
  'foundations/module-00-genai-foundations-from-nlp-to-transformers/index.html',
  'foundations/module-01-ai-engineering-essentials/exercises/index.html',
];

const bad = [];

for (const rel of required) {
  const candidates = Array.isArray(rel) ? rel : [rel];
  if (!candidates.some((c) => fs.existsSync(path.join(SITE, c)))) {
    bad.push(`missing: ${candidates.join(' or ')}`);
  }
}

function scanHtml(dir, rel = '') {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const r = rel ? `${rel}/${name}` : name;
    if (fs.statSync(full).isDirectory()) scanHtml(full, r);
    else if (name.endsWith('.html')) {
      const html = fs.readFileSync(full, 'utf-8');
      const matches = html.matchAll(/href="([^"]+\.md)"/g);
      for (const [, href] of matches) {
        if (!href.startsWith('http') && !href.includes('github.com')) {
          bad.push(`${r}: href="${href}"`);
        }
      }
    }
  }
}

if (!fs.existsSync(SITE)) {
  console.error('site/ not found');
  process.exit(1);
}

scanHtml(SITE);

if (bad.length) {
  console.error('FAILED:\n', bad.slice(0, 20).join('\n'));
  process.exit(1);
}
console.log(`OK: ${required.length} required pages, no internal .md hrefs`);
