#!/usr/bin/env node
/**
 * Import from ai-learning platform into organized handbook structure.
 *
 * Layout:
 *   docs/{phase}/module-{NN}-{slug}/
 *     index.md          — module overview + lesson table
 *     lessons/*.md      — lesson content
 *     exercises/*       — starter/solution py (+ ipynb)
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
  'module-00': { phase: 'foundations', order: 1 },
  'module-01': { phase: 'foundations', order: 2 },
  'module-05': { phase: 'foundations', order: 3 },
  'module-06': { phase: 'foundations', order: 4 },
  'module-07': { phase: 'foundations', order: 5 },
  'module-09': { phase: 'build', order: 1 },
  'module-11': { phase: 'build', order: 2 },
  'module-12': { phase: 'build', order: 3 },
  'module-13': { phase: 'build', order: 4 },
  'module-14': { phase: 'build', order: 5 },
  'module-18': { phase: 'build', order: 6, handbookNative: true },
  'module-10': { phase: 'production', order: 1 },
  'module-16': { phase: 'production', order: 2 },
  'module-19': { phase: 'production', order: 3, handbookNative: true },
  'module-15': { phase: 'advanced', order: 1 },
  'module-17': { phase: 'advanced', order: 2 },
};

/** Handbook-only modules — never deleted or overwritten by platform import */
const HANDBOOK_NATIVE_IDS = new Set(
  Object.entries(MODULE_PHASE)
    .filter(([, m]) => m.handbookNative)
    .map(([id]) => id),
);

const PHASE_META = {
  foundations: { label: 'Foundations', summary: 'Core ML, transformers, and LLMs.' },
  build: { label: 'Build', summary: 'RAG, agents, harness, tools, and prompts.' },
  production: { label: 'Production', summary: 'LLMOps, evals, deployment, and safety.' },
  advanced: { label: 'Advanced', summary: 'Fine-tuning and capstone projects.' },
};

const PHASE_ORDER = ['foundations', 'build', 'production', 'advanced'];

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 55);
}

function moduleFolderName(mod) {
  const num = mod.id.replace('module-', '');
  return `module-${num}-${slugify(mod.title)}`;
}

function moduleLabel(mod) {
  const num = mod.id.replace('module-', '').padStart(2, '0');
  return `M${num} · ${mod.title}`;
}

function readModule(moduleId) {
  if (!MODULE_PHASE[moduleId]) return null;
  if (HANDBOOK_NATIVE_IDS.has(moduleId)) {
    return readHandbookNativeModule(moduleId);
  }
  const modulePath = path.join(SOURCE, moduleId);
  const jsonPath = path.join(modulePath, 'module.json');
  if (!fs.existsSync(jsonPath)) return null;
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
  if (data.listed === false) return null;
  return { ...data, modulePath, meta: MODULE_PHASE[moduleId], handbookNative: false };
}

function readHandbookNativeModule(moduleId) {
  const phase = MODULE_PHASE[moduleId].phase;
  const dirs = fs.readdirSync(path.join(DOCS, phase)).filter((d) => d.startsWith(`${moduleId}-`));
  if (dirs.length === 0) return null;
  const folder = dirs[0];
  const moduleDir = path.join(DOCS, phase, folder);
  const indexPath = path.join(moduleDir, 'index.md');
  if (!fs.existsSync(indexPath)) return null;
  const parsed = matter(fs.readFileSync(indexPath, 'utf-8'));
  return {
    id: moduleId,
    title: parsed.data.title || moduleId,
    description: parsed.content.split('\n')[0] || '',
    estimatedHours: countLessonsInDir(moduleDir) * 0.6,
    modulePath: moduleDir,
    meta: MODULE_PHASE[moduleId],
    handbookNative: true,
  };
}

function countLessonsInDir(moduleDir) {
  const lessonsDir = path.join(moduleDir, 'lessons');
  if (!fs.existsSync(lessonsDir)) return 0;
  return fs.readdirSync(lessonsDir).filter((f) => f.endsWith('.md')).length;
}

function listLessons(modulePath) {
  const lessonsDir = path.join(modulePath, 'lessons');
  if (!fs.existsSync(lessonsDir)) return [];
  return fs.readdirSync(lessonsDir).filter((f) => f.endsWith('.md')).sort();
}

function copyExercise(lessonsDir, lessonId, destExercises) {
  const codeDir = path.join(lessonsDir, lessonId);
  if (!fs.existsSync(codeDir)) return false;
  const starter = path.join(codeDir, 'starter.py');
  const solution = path.join(codeDir, 'solution.py');
  if (!fs.existsSync(starter) && !fs.existsSync(solution)) return false;

  fs.mkdirSync(destExercises, { recursive: true });
  const base = lessonId.replace(/^lesson-/, '');
  if (fs.existsSync(starter)) fs.copyFileSync(starter, path.join(destExercises, `${base}-starter.py`));
  if (fs.existsSync(solution)) fs.copyFileSync(solution, path.join(destExercises, `${base}-solution.py`));
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
    module: moduleMeta.id,
  };
  if (parsed.data.youtubeUrl) fm.youtube = parsed.data.youtubeUrl;
  if (parsed.data.tests?.length) fm.objectives = parsed.data.tests;
  return matter.stringify(parsed.content.trim(), fm);
}

function clearPlatformModules() {
  for (const moduleId of Object.keys(MODULE_PHASE)) {
    if (HANDBOOK_NATIVE_IDS.has(moduleId)) continue;
    const mod = readModuleFromPlatform(moduleId);
    if (!mod) continue;
    const folder = moduleFolderName(mod);
    const dir = path.join(DOCS, mod.meta.phase, folder);
    if (fs.existsSync(dir)) {
      if (moduleId === 'module-12') {
        clearModuleKeepingHandbookLessons(dir);
      } else {
        fs.rmSync(dir, { recursive: true });
      }
    }
  }
}

function readModuleFromPlatform(moduleId) {
  const modulePath = path.join(SOURCE, moduleId);
  const jsonPath = path.join(modulePath, 'module.json');
  if (!fs.existsSync(jsonPath)) return null;
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
  if (data.listed === false) return null;
  if (!MODULE_PHASE[moduleId]) return null;
  return { ...data, modulePath, meta: MODULE_PHASE[moduleId] };
}

/** Keep handbook-authored lessons (e.g. M12 L04–10) when re-importing from platform */
function clearModuleKeepingHandbookLessons(moduleDir) {
  const lessonsDir = path.join(moduleDir, 'lessons');
  const preserved = new Map();
  if (fs.existsSync(lessonsDir)) {
    for (const file of fs.readdirSync(lessonsDir)) {
      if (!file.endsWith('.md')) continue;
      const num = parseInt(file.slice(0, 2), 10);
      if (num >= 4) preserved.set(file, fs.readFileSync(path.join(lessonsDir, file), 'utf-8'));
    }
  }
  fs.rmSync(moduleDir, { recursive: true });
  if (preserved.size > 0) {
    fs.mkdirSync(lessonsDir, { recursive: true });
    for (const [file, content] of preserved) {
      fs.writeFileSync(path.join(lessonsDir, file), content);
    }
  }
}

function buildLessonTable(lessons) {
  const rows = lessons.map((l, i) => {
    const link = `lessons/${l.file}`;
    return `| ${i + 1} | [${l.title}](${link}) | ${l.duration || '—'} | ${l.difficulty || '—'} |`;
  });
  return [
    '| # | Lesson | Duration | Level |',
    '|---|--------|----------|-------|',
    ...rows,
    '',
  ].join('\n');
}

function main() {
  if (!fs.existsSync(SOURCE)) {
    console.error(`Source not found: ${SOURCE}`);
    process.exit(1);
  }

  clearPlatformModules();

  const modules = Object.keys(MODULE_PHASE)
    .map(readModule)
    .filter(Boolean)
    .sort((a, b) => {
      const phaseDiff = PHASE_ORDER.indexOf(a.meta.phase) - PHASE_ORDER.indexOf(b.meta.phase);
      if (phaseDiff !== 0) return phaseDiff;
      return a.meta.order - b.meta.order;
    });

  const navByPhase = Object.fromEntries(PHASE_ORDER.map((p) => [p, []]));
  const learningPathRows = [];
  const counters = { totalLessons: 0, totalExercises: 0 };

  for (const mod of modules) {
    const phase = mod.meta.phase;
    const folder = moduleFolderName(mod);
    const relPath = `${phase}/${folder}`;
    const moduleDir = path.join(DOCS, relPath);
    const lessonsDir = path.join(moduleDir, 'lessons');
    const exercisesDir = path.join(moduleDir, 'exercises');

    if (mod.handbookNative) {
      registerNativeModule(mod, navByPhase, learningPathRows, counters);
      continue;
    }

    const srcLessonsDir = path.join(mod.modulePath, 'lessons');
    fs.mkdirSync(lessonsDir, { recursive: true });

    const lessonMeta = [];
    for (const file of listLessons(mod.modulePath)) {
      const raw = fs.readFileSync(path.join(srcLessonsDir, file), 'utf-8');
      const parsed = matter(raw);
      const lessonId = parsed.data.id || file.replace('.md', '');

      fs.writeFileSync(
        path.join(lessonsDir, file),
        transformLesson(raw, { id: mod.id, title: mod.title, phaseLabel: PHASE_META[phase].label }),
      );

      if (copyExercise(srcLessonsDir, lessonId, exercisesDir)) counters.totalExercises++;

      lessonMeta.push({
        file,
        title: parsed.data.title || lessonId,
        duration: parsed.data.duration,
        difficulty: parsed.data.difficulty,
      });
      counters.totalLessons++;
    }

    if (mod.id === 'module-12') {
      const before = lessonMeta.length;
      mergeHandbookLessons(lessonsDir, lessonMeta);
      counters.totalLessons += lessonMeta.length - before;
    }

    const lessonTable = buildLessonTable(lessonMeta);
    const indexBody = [
      mod.description,
      '',
      `| | |`,
      `|---|---|`,
      `| **Module ID** | \`${mod.id}\` |`,
      `| **Phase** | ${PHASE_META[phase].label} |`,
      `| **Lessons** | ${lessonMeta.length} |`,
      `| **Est. hours** | ~${mod.estimatedHours}h |`,
      '',
      '## Lessons',
      '',
      lessonTable,
      lessonMeta.length > 0
        ? `\n**Start here:** [${lessonMeta[0].title}](lessons/${lessonMeta[0].file})\n`
        : '',
      exercisesDir && fs.existsSync(exercisesDir)
        ? '## Exercises\n\nPython files are in the [`exercises/`](exercises/) folder (`*-starter.py` and `*-solution.py`).\n'
        : '',
    ]
      .filter(Boolean)
      .join('\n');

    fs.writeFileSync(
      path.join(moduleDir, 'index.md'),
      matter.stringify(indexBody, {
        title: mod.title,
        module_id: mod.id,
        phase: PHASE_META[phase].label,
      }),
    );

    navByPhase[phase].push({
      label: moduleLabel(mod),
      path: `${relPath}/index.md`,
    });

    learningPathRows.push({
      phase: PHASE_META[phase].label,
      num: mod.id.replace('module-', '').padStart(2, '0'),
      title: mod.title,
      lessons: lessonMeta.length,
      hours: mod.estimatedHours,
      path: `${relPath}/index.md`,
      partial: false,
    });
  }

  writePhaseIndexes(navByPhase);
  writeLearningPath(learningPathRows);
  updateMkdocs(buildNavYaml(navByPhase));

  console.log(`Migrated ${modules.length} modules, ${counters.totalLessons} lessons, ${counters.totalExercises} exercise sets.`);
}

function writePhaseIndexes(navByPhase) {
  for (const phase of PHASE_ORDER) {
    const meta = PHASE_META[phase];
    const modules = navByPhase[phase];
    const rows = modules.map(
      (m, i) => `| ${i + 1} | [${m.label}](${m.path.replace(/^[^/]+\//, '')}) |`,
    );
    const body = [
      meta.summary,
      '',
      '| # | Module |',
      '|---|--------|',
      ...rows,
      '',
    ].join('\n');
    fs.writeFileSync(
      path.join(DOCS, phase, 'index.md'),
      matter.stringify(body, { title: meta.label }),
    );
  }
}

function writeLearningPath(rows) {
  let currentPhase = '';
  const lines = [
    'Four phases · Sixteen modules · Follow in order',
    '',
  ];
  for (const row of rows) {
    if (row.phase !== currentPhase) {
      currentPhase = row.phase;
      lines.push(`## ${currentPhase}`, '', '| Module | Title | Lessons | Hours |', '|---|-------|---------|-------|');
    }
    const partial = row.partial ? ' *(partial)*' : '';
    lines.push(`| [M${row.num}](${row.path}) | ${row.title}${partial} | ${row.lessons} | ~${row.hours}h |`);
  }
  lines.push('', '---', '', 'Use phase overviews:', '');
  for (const phase of PHASE_ORDER) {
    lines.push(`- [${PHASE_META[phase].label}](${phase}/index.md)`);
  }

  fs.writeFileSync(
    path.join(DOCS, 'learning-path.md'),
    matter.stringify(lines.join('\n'), { title: 'Learning Path' }),
  );
}

function mergeHandbookLessons(lessonsDir, lessonMeta) {
  if (!fs.existsSync(lessonsDir)) return;
  const existing = new Set(lessonMeta.map((l) => l.file));
  for (const file of fs.readdirSync(lessonsDir).filter((f) => f.endsWith('.md')).sort()) {
    if (existing.has(file)) continue;
    const raw = fs.readFileSync(path.join(lessonsDir, file), 'utf-8');
    const parsed = matter(raw);
    lessonMeta.push({
      file,
      title: parsed.data.title || file.replace('.md', ''),
      duration: parsed.data.duration,
      difficulty: parsed.data.difficulty,
    });
    existing.add(file);
  }
  lessonMeta.sort((a, b) => a.file.localeCompare(b.file));
}

function nativeModuleRelPath(moduleId) {
  const phase = MODULE_PHASE[moduleId].phase;
  const dirs = fs.readdirSync(path.join(DOCS, phase)).filter((d) => d.startsWith(`${moduleId}-`));
  if (dirs.length === 0) return null;
  return `${phase}/${dirs[0]}`;
}

function registerNativeModule(mod, navByPhase, learningPathRows, counters) {
  const relPath = nativeModuleRelPath(mod.id);
  if (!relPath) return;
  const lessons = countLessonsInDir(path.join(DOCS, relPath));
  counters.totalLessons += lessons;
  navByPhase[mod.meta.phase].push({
    label: moduleLabel(mod),
    path: `${relPath}/index.md`,
  });
  learningPathRows.push({
    phase: PHASE_META[mod.meta.phase].label,
    num: mod.id.replace('module-', '').padStart(2, '0'),
    title: mod.title,
    lessons,
    hours: Math.round(lessons * 0.6 * 10) / 10,
    path: `${relPath}/index.md`,
    partial: false,
  });
}

function buildNavYaml(navByPhase) {
  const lines = [
    'nav:',
    '  - Home: index.md',
    '  - Getting Started: getting-started.md',
    '  - Topic Map: topic-map.md',
    '  - Learning Path: learning-path.md',
    '  - Glossary: glossary.md',
    '  - Agentic AI: agentic-ai/index.md',
    '  - Evals & Observability: evals-observability/index.md',
  ];
  for (const phase of PHASE_ORDER) {
    lines.push(`  - ${PHASE_META[phase].label}:`);
    lines.push(`      - Overview: ${phase}/index.md`);
    for (const mod of navByPhase[phase]) {
      lines.push(`      - ${escapeYaml(mod.label)}: ${mod.path}`);
    }
  }
  lines.push('  - Resources:');
  lines.push('      - Overview: resources/index.md');
  lines.push('      - Essential Papers: resources/essential-papers.md');
  lines.push('      - Essential Videos: resources/essential-videos.md');
  lines.push('      - Open Source Hubs: resources/open-source-hubs.md');
  lines.push('      - Courses & Communities: resources/courses-and-communities.md');
  lines.push('      - All Papers: resources/papers.md');
  lines.push('      - All Videos: resources/videos.md');
  lines.push('      - Tools & Libraries: resources/tools-and-libraries.md');
  lines.push('  - Exercises: exercises/index.md');
  lines.push('  - Projects: projects/index.md');
  lines.push('  - Roadmap: roadmap.md');
  lines.push('  - Contribute: contribute.md');
  return lines.join('\n');
}

function escapeYaml(s) {
  if (/[:#\[\]{}|>&*!%@`]/.test(s) || s.includes('"')) {
    return `"${s.replace(/"/g, '\\"')}"`;
  }
  return s;
}

function updateMkdocs(navYaml) {
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
    - navigation.sections
    - navigation.indexes
    - search.suggest
    - search.highlight
    - content.code.copy
    - toc.integrate

plugins:
  - search

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - tables
  - toc:
      permalink: true

extra_javascript:
  - https://unpkg.com/mermaid@10/dist/mermaid.min.js

validation:
  omitted_files: ignore
  absolute_links: ignore
  unrecognized_links: ignore

`;
  fs.writeFileSync(path.join(ROOT, 'mkdocs.yml'), `${base}${navYaml}\n`);
}

main();
