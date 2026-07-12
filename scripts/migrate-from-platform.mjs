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
  'module-10': { phase: 'production', order: 1 },
  'module-16': { phase: 'production', order: 2 },
  'module-15': { phase: 'advanced', order: 1 },
  'module-17': { phase: 'advanced', order: 2 },
};

const PHASE_META = {
  foundations: { label: 'Foundations', summary: 'Core ML, transformers, and LLMs.' },
  build: { label: 'Build', summary: 'RAG, agents, vector search, and prompts.' },
  production: { label: 'Production', summary: 'LLMOps, deployment, and safety.' },
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
  const modulePath = path.join(SOURCE, moduleId);
  const jsonPath = path.join(modulePath, 'module.json');
  if (!fs.existsSync(jsonPath)) return null;
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
  if (data.listed === false) return null;
  if (!MODULE_PHASE[moduleId]) return null;
  return { ...data, modulePath, meta: MODULE_PHASE[moduleId] };
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

function clearPhaseDirs() {
  for (const phase of PHASE_ORDER) {
    const dir = path.join(DOCS, phase);
    if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true });
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

  clearPhaseDirs();

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
  let totalLessons = 0;
  let totalExercises = 0;

  for (const mod of modules) {
    const phase = mod.meta.phase;
    const folder = moduleFolderName(mod);
    const relPath = `${phase}/${folder}`;
    const moduleDir = path.join(DOCS, relPath);
    const lessonsDir = path.join(moduleDir, 'lessons');
    const exercisesDir = path.join(moduleDir, 'exercises');
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

      if (copyExercise(srcLessonsDir, lessonId, exercisesDir)) totalExercises++;

      lessonMeta.push({
        file,
        title: parsed.data.title || lessonId,
        duration: parsed.data.duration,
        difficulty: parsed.data.difficulty,
      });
      totalLessons++;
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
      mod.id === 'module-12' ? `| **Status** | Partial — see [GAPS.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/GAPS.md) |` : '',
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
      partial: mod.id === 'module-12',
    });
  }

  writePhaseIndexes(navByPhase);
  writeLearningPath(learningPathRows);
  updateMkdocs(buildNavYaml(navByPhase));

  console.log(`Migrated ${modules.length} modules, ${totalLessons} lessons, ${totalExercises} exercise sets.`);
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
    'Four phases · Fourteen modules · Follow in order',
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

function buildNavYaml(navByPhase) {
  const lines = [
    'nav:',
    '  - Home: index.md',
    '  - Learning Path: learning-path.md',
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
  lines.push('      - Papers: resources/papers.md');
  lines.push('      - Videos: resources/videos.md');
  lines.push('      - Tools & Libraries: resources/tools-and-libraries.md');
  lines.push('  - Projects: projects/index.md');
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
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - tables
  - toc:
      permalink: true

`;
  fs.writeFileSync(path.join(ROOT, 'mkdocs.yml'), `${base}${navYaml}\n`);
}

main();
