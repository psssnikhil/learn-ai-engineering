#!/usr/bin/env node
/**
 * Create exercises/index.md for each module with an exercises/ folder.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(__dirname, '..', 'docs');

function walk(dir, out = []) {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) {
      if (name === 'exercises') out.push(full);
      else walk(full, out);
    }
  }
  return out;
}

for (const exDir of walk(DOCS)) {
  const files = fs.readdirSync(exDir).filter((f) => f.endsWith('.py') || f.endsWith('.ipynb'));
  if (files.length === 0) continue;

  const starters = files.filter((f) => f.includes('starter')).sort();
  const solutions = files.filter((f) => f.includes('solution')).sort();
  const moduleDir = path.dirname(exDir);
  const relToModule = path.basename(exDir);

  const lines = [
    '---',
    'title: Exercises',
    '---',
    '',
    '# Exercises',
    '',
    'Run from this folder in your terminal. Compare your work with `*-solution.py` when stuck.',
    '',
    '```bash',
    `cd ${relToModule}`,
    'python 01-starter.py   # use the starter file name below',
    '```',
    '',
    '## Starter files',
    '',
    ...starters.map((f) => `- \`${f}\``),
    '',
  ];

  if (solutions.length) {
    lines.push('## Solutions', '', ...solutions.map((f) => `- \`${f}\``), '');
  }

  const ipynb = files.filter((f) => f.endsWith('.ipynb'));
  if (ipynb.length) {
    lines.push('## Notebooks', '', ...ipynb.map((f) => `- \`${f}\``), '');
  }

  lines.push(
    'Back to [course overview](../index.md).',
    '',
  );

  fs.writeFileSync(path.join(exDir, 'index.md'), lines.join('\n'));
  console.log('Created', path.relative(DOCS, path.join(exDir, 'index.md')));
}
