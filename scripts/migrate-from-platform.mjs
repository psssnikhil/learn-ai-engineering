#!/usr/bin/env node
/**
 * One-time import from ai-learning platform content.
 * Source: ../ai-learning/website/content/modules
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const SOURCE = path.resolve(ROOT, '../ai-learning/website/content/modules');
const DOCS = path.join(ROOT, 'docs');

const MODULE_PHASE = {
  'module-00': 'foundations',
  'module-01': 'foundations',
  'module-05': 'foundations',
  'module-06': 'foundations',
  'module-07': 'foundations',
  'module-09': 'build',
  'module-11': 'build',
  'module-12': 'build',
  'module-13': 'build',
  'module-14': 'build',
  'module-10': 'production',
  'module-16': 'production',
  'module-15': 'advanced',
  'module-17': 'advanced',
};

const PHASE_LABELS = {
  foundations: 'Foundations',
  build: 'Build',
  production: 'Production',
  advanced: 'Advanced',
};

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 48);
}

function lessonFileName(file) {
  return file.replace(/\.md$/, '').replace(/^\d+-/, '') || file;
}

function readModule(moduleId) {
  const modulePath = path.join(SOURCE, moduleId);
  const jsonPath = path.join(modulePath, 'module.json');
  if (!fs.existsSync(jsonPath)) return null;
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
  if (data.listed === false) return null;
  if (!MODULE_PHASE[moduleId]) return null;
  return { ...data, modulePath };
}

function listLessons(modulePath) {
  const lessonsDir = path.join(modulePath, 'lessons');
  if (!fs.existsSync(lessonsDir)) return [];
  return fs
    .readdirSync(lessonsDir)
    .filter((f) => f.endsWith('.md'))
    .sort();
}

function copyExercise(lessonsDir, lessonId, destExercises) {
  const codeDir = path.join(lessonsDir, lessonId);
  if (!fs.existsSync(codeDir)) return false;
  const starter = path.join(codeDir, 'starter.py');
  const solution = path.join(codeDir, 'solution.py');
  if (!fs.existsSync(starter) && !fs.existsSync(solution)) return false;

  fs.mkdirSync(destExercises, { recursive: true });
  const base = lessonId.replace(/^lesson-/, '');
  if (fs.existsSync(starter)) {
    fs.copyFileSync(starter, path.join(destExercises, `${base}-starter.py`));
  }
  if (fs.existsSync(solution)) {
    fs.copyFileSync(solution, path.join(destExercises, `${base}-solution.py`));
  }
  return true;
}

function transformLesson(content, moduleMeta) {
  const parsed = matter(content);
  const fm = {
    title: parsed.data.title || 'Untitled',
    description: parsed.data.description || '',
    duration: parsed.data.duration || '',
    difficulty: parsed.data.difficulty || 'beginner',
    has_code: parsed.data.hasCode ?? false,
  };
  if (parsed.data.youtubeUrl) fm.youtube = parsed.data.youtubeUrl;
  if (parsed.data.tests?.length) fm.objectives = parsed.data.tests;

  const header = `> **Module:** ${moduleMeta.title} · **Phase:** ${moduleMeta.phaseLabel}\n\n`;
  return matter.stringify(parsed.content.trim(), fm);
}

function clearGeneratedDocs() {
  for (const phase of Object.keys(PHASE_LABELS)) {
    const phaseDir = path.join(DOCS, phase);
    if (fs.existsSync(phaseDir)) {
      fs.rmSync(phaseDir, { recursive: true });
    }
  }
}

function main() {
  if (!fs.existsSync(SOURCE)) {
    console.error(`Source not found: ${SOURCE}`);
    process.exit(1);
  }

  clearGeneratedDocs();

  const modules = Object.keys(MODULE_PHASE)
    .map(readModule)
    .filter(Boolean)
    .sort((a, b) => a.order - b.order);

  const nav = {
    Home: 'index.md',
    'Learning Path': 'learning-path.md',
    Resources: 'resources/index.md',
    Projects: 'projects/index.md',
  };

  let totalLessons = 0;
  let totalExercises = 0;

  for (const mod of modules) {
    const phase = MODULE_PHASE[mod.id];
    const phaseLabel = PHASE_LABELS[phase];
    const folderName = `${String(mod.order).padStart(2, '0')}-${slugify(mod.title)}`;
    const moduleDir = path.join(DOCS, phase, folderName);
    const exercisesDir = path.join(moduleDir, 'exercises');
    const lessonsDir = path.join(mod.modulePath, 'lessons');

    fs.mkdirSync(moduleDir, { recursive: true });

    const indexMd = matter.stringify(
      `${mod.description}\n\n## Lessons\n\nBrowse the lessons in the sidebar, or open the first lesson to begin.\n`,
      {
        title: mod.title,
        phase: phaseLabel,
        estimated_hours: mod.estimatedHours,
        module_order: mod.order,
        status: mod.status || 'active',
      },
    );
    fs.writeFileSync(path.join(moduleDir, 'index.md'), indexMd);

    const lessonNav = [];
    const lessonFiles = listLessons(mod.modulePath);

    for (const file of lessonFiles) {
      const raw = fs.readFileSync(path.join(lessonsDir, file), 'utf-8');
      const parsed = matter(raw);
      const lessonId = parsed.data.id || lessonFileName(file);
      const outName = `${file}`;
      const outPath = path.join(moduleDir, outName);

      fs.writeFileSync(
        outPath,
        transformLesson(raw, { title: mod.title, phaseLabel }),
      );

      if (copyExercise(lessonsDir, lessonId, exercisesDir)) {
        totalExercises++;
      }

      lessonNav.push({ title: parsed.data.title || lessonId, path: `${phase}/${folderName}/${outName}` });
      totalLessons++;
    }

    if (!nav[phaseLabel]) nav[phaseLabel] = [];
    const moduleEntry = {
      title: mod.title,
      overview: `${phase}/${folderName}/index.md`,
      lessons: lessonNav,
    };
    nav[phaseLabel].push(moduleEntry);
  }

  const navYaml = buildNavYaml(nav);
  updateMkdocs(navYaml);

  console.log(`Migrated ${modules.length} modules, ${totalLessons} lessons, ${totalExercises} exercise sets.`);
}

function buildNavYaml(nav) {
  const lines = ['nav:'];
  for (const [key, value] of Object.entries(nav)) {
    if (typeof value === 'string') {
      lines.push(`  - ${escapeYaml(key)}: ${value}`);
      continue;
    }
    if (!Array.isArray(value)) continue;
    lines.push(`  - ${escapeYaml(key)}:`);
    for (const mod of value) {
      lines.push(`      - ${escapeYaml(mod.title)}:`);
      lines.push(`          - Overview: ${mod.overview}`);
      for (const lesson of mod.lessons) {
        lines.push(`          - ${escapeYaml(lesson.title)}: ${lesson.path}`);
      }
    }
  }
  return lines.join('\n');
}

function escapeYaml(s) {
  if (/[:#\[\]{}|>&*!%@`]/.test(s) || s.includes('"')) {
    return `"${s.replace(/"/g, '\\"')}"`;
  }
  return s;
}

function updateMkdocs(navYaml) {
  const mkdocsPath = path.join(ROOT, 'mkdocs.yml');
  const base = `site_name: AI Engineering Handbook
site_description: Open-source knowledge base for learning AI engineering — agents, RAG, LLMOps, and more.
site_url: https://psssnikhil.github.io/learn-ai-engineering/
repo_url: https://github.com/psssnikhil/learn-ai-engineering
repo_name: learn-ai-engineering

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Light mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.expand
    - navigation.sections
    - search.suggest
    - search.highlight
    - content.code.copy

plugins:
  - search

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - tables
  - toc:
      permalink: true

`;
  fs.writeFileSync(mkdocsPath, `${base}${navYaml}\n`);
}

main();
