#!/usr/bin/env node
/**
 * Normalize module index.md for MkDocs — fix metadata rendering, learner-facing copy.
 * Run: node scripts/fix-module-indexes.mjs
 */
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(__dirname, '..', 'docs');

const COURSE_NUM = {
  'module-00': '01', 'module-01': '02', 'module-05': '03', 'module-06': '04',
  'module-07': '05', 'module-09': '06', 'module-11': '07', 'module-18': '08',
  'module-12': '09', 'module-13': '10', 'module-14': '11', 'module-10': '12',
  'module-19': '13', 'module-16': '14', 'module-15': '15', 'module-17': '16',
};

const HOURS = {
  'module-00': 15, 'module-01': 8, 'module-05': 20, 'module-06': 18, 'module-07': 22,
  'module-09': 18, 'module-11': 16, 'module-18': 10, 'module-12': 14, 'module-13': 12,
  'module-14': 12, 'module-10': 16, 'module-19': 10, 'module-16': 12, 'module-15': 14,
  'module-17': 20,
};

function moduleIdFromPath(filePath) {
  const folder = path.basename(path.dirname(filePath));
  const m = folder.match(/^(module-\d+)/);
  return m ? m[1] : '';
}

function parseLessons(body) {
  const m = body.match(/## Lessons[\s\S]*?(?=\n## |\n\*\*Start here|\Z)/);
  return m ? m[0].trim() : '';
}

function extractDescription(content) {
  let text = content
    .replace(/^#\s+.+\n+/m, '')
    .replace(/\*\*Course \d+\*\*[^\n]+\n+/g, '')
    .replace(/\|[^|\n]+\|[^|\n]*\|[^|\n]*\|/g, '')
    .replace(/## Lessons[\s\S]*/, '')
    .replace(/## Exercises[\s\S]*/, '')
    .replace(/\*\*Start here:\*\*[\s\S]*/, '')
    .trim();
  return text.split('\n\n')[0]?.trim() || '';
}

function rebuildIndex(filePath) {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const { data, content } = matter(raw);
  const moduleId = data.module_id || moduleIdFromPath(filePath);
  const courseNum = COURSE_NUM[moduleId] || '';
  const phase = data.phase || '';
  const title = data.title || 'Course';

  const description = extractDescription(content);
  let lessonsBlock = parseLessons(content);
  const lessonCount = (lessonsBlock.match(/\| \d+ \|/g) || []).length;
  const hours = HOURS[moduleId] || '?';

  const metaLine = courseNum
    ? `**Course ${courseNum}** · ${phase} · ${lessonCount} lessons · ~${hours}h`
    : `${phase} · ${lessonCount} lessons · ~${hours}h`;

  const startMatch = lessonsBlock.match(/\| 1 \| \[([^\]]+)\]\(([^)]+)\)/);
  const startHere = startMatch
    ? `\n**Start here:** [${startMatch[1]}](${startMatch[2]})\n`
    : '';

  const exercisesLink = fs.existsSync(path.join(path.dirname(filePath), 'exercises'))
    ? `\n## Exercises\n\nHands-on files: [exercises/index.md](exercises/index.md)\n`
    : '';

  const body = [
    `# ${title}`,
    '',
    description,
    '',
    metaLine,
    '',
    lessonsBlock.replace(/^## Lessons\n?/, '## Lessons\n\n'),
    startHere.trim(),
    exercisesLink.trim(),
  ]
    .filter(Boolean)
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim() + '\n';

  const fm = { ...data, module_id: moduleId };
  fs.writeFileSync(filePath, matter.stringify(body, fm));
}

for (const p of findModuleIndexes(DOCS)) {
  rebuildIndex(p);
  console.log('Fixed', path.relative(DOCS, p));
}

function findModuleIndexes(dir, out = []) {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (!fs.statSync(full).isDirectory()) continue;
    if (name.startsWith('module-')) {
      const idx = path.join(full, 'index.md');
      if (fs.existsSync(idx)) out.push(idx);
    } else findModuleIndexes(full, out);
  }
  return out;
}

console.log('Done.');
