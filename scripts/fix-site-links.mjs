#!/usr/bin/env node
/**
 * Fix raw .md hrefs in built HTML for GitHub Pages pretty URLs.
 * MkDocs transforms markdown links but not raw HTML href="*.md" in md_in_html blocks.
 *
 * Run after: mkdocs build
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.join(__dirname, '..', 'site');

function mdHrefToSitePath(href) {
  if (!href.endsWith('.md') || href.startsWith('http') || href.startsWith('//')) {
    return href;
  }
  let p = href.slice(0, -3);
  if (p.endsWith('/index')) {
    p = p.slice(0, -'/index'.length);
  }
  return p.endsWith('/') ? p : `${p}/`;
}

function fixHtml(html) {
  let count = 0;
  const fixed = html.replace(/href="([^"]+\.md)"/g, (_, href) => {
    count += 1;
    return `href="${mdHrefToSitePath(href)}"`;
  });
  return { fixed, count };
}

function walk(dir) {
  let total = 0;
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) {
      total += walk(full);
    } else if (name.endsWith('.html')) {
      const html = fs.readFileSync(full, 'utf-8');
      const { fixed, count } = fixHtml(html);
      if (count > 0) {
        fs.writeFileSync(full, fixed);
        total += count;
      }
    }
  }
  return total;
}

if (!fs.existsSync(SITE)) {
  console.error('site/ not found — run mkdocs build first');
  process.exit(1);
}

const fixed = walk(SITE);
console.log(`Fixed ${fixed} raw .md href(s) in ${SITE}`);

// Fail CI if any internal .md hrefs remain (ignore github.com edit links)
const remaining = [];
function scan(dir, rel = '') {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const r = rel ? `${rel}/${name}` : name;
    if (fs.statSync(full).isDirectory()) scan(full, r);
    else if (name.endsWith('.html')) {
      const matches = fs.readFileSync(full, 'utf-8').matchAll(/href="([^"]+\.md)"/g);
      for (const [, href] of matches) {
        if (!href.startsWith('http') && !href.startsWith('//')) {
          remaining.push(`${r}: href="${href}"`);
        }
      }
    }
  }
}
scan(SITE);
if (remaining.length) {
  console.error('Remaining .md hrefs:', remaining.slice(0, 10).join('\n'));
  process.exit(1);
}
