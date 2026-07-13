#!/usr/bin/env node
/**
 * Generate mkdocs.yml nav from curriculum.yml
 *
 * Navigation model:
 *   - One sequential "Learn" tab (course index pages only — lessons live inside each course)
 *   - Optional tracks at the end of Learn
 *   - Reference + Projects tabs
 *
 * Run: npm run sync-nav
 * Also runs automatically after: npm run migrate
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const DOCS = path.join(ROOT, 'docs');

function loadCurriculum() {
  const raw = fs.readFileSync(path.join(ROOT, 'curriculum.yml'), 'utf-8');
  return yaml.parse(raw);
}

function escapeYaml(s) {
  if (/[:#\[\]{}|>&*!%@`]/.test(s) || s.includes('"')) {
    return `"${s.replace(/"/g, '\\"')}"`;
  }
  return s;
}

function prefixCourse(num, title) {
  const n = String(num).padStart(2, '0');
  return `${n}. ${title}`;
}

function buildNavYaml(curriculum) {
  const lines = ['nav:'];
  lines.push('  - Home: index.md');
  lines.push('  - Start Here: start-here.md');
  lines.push('  - Learn:');
  lines.push('      - Overview: learn/index.md');

  curriculum.courses.forEach((course, i) => {
    const label = prefixCourse(i + 1, course.title);
    lines.push(`      - ${escapeYaml(label)}: ${course.path}/index.md`);
  });

  for (const track of curriculum.tracks) {
    lines.push(`      - ${escapeYaml(`Track · ${track.title}`)}:`);
    for (const page of track.pages) {
      lines.push(`          - ${escapeYaml(page.title)}: ${track.path}/${page.file}`);
    }
  }

  lines.push('  - Reference:');
  for (const item of curriculum.reference) {
    lines.push(`      - ${escapeYaml(item.title)}: ${item.path}`);
  }

  lines.push('  - Projects:');
  for (const item of curriculum.projects) {
    lines.push(`      - ${escapeYaml(item.title)}: ${item.path}`);
  }

  lines.push('  - Contribute: contribute.md');
  return lines.join('\n');
}

function courseHref(coursePath) {
  return `../${coursePath}/index.md`;
}

function writeLearnIndex(curriculum) {
  const rows = curriculum.courses.map((course, i) => {
    const n = i + 1;
    const href = courseHref(course.path);
    return `| ${n} | [${course.title}](${href}) | [Start →](${href}) |`;
  });

  const trackRows = curriculum.tracks.map(
    (t) => `| [${t.title}](../${t.path}/index.md) | Optional focused path |`,
  );

  const body = `Follow this order from top to bottom. Each course opens with a lesson list — work through them in sequence.

## Core path (${curriculum.courses.length} courses)

| # | Course | |
|---|--------|---|
${rows.join('\n')}

## Optional tracks

Read these when you need a focused path on agents or modern tooling:

| Track | |
|-------|---|
${trackRows.join('\n')}

## Quick links

- [Start Here](../start-here.md) — pick a path by background
- [FAQ](../faq.md) — RAG vs fine-tune vs agents
- [Build These First](../projects/build-these.md) — portfolio projects
- [Topic Map](../topic-map.md) — find any concept

## Adding new content

Contributors: edit \`curriculum.yml\` at the repo root, add your course in order, run \`npm run sync-nav\`. See [CONTRIBUTING.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/CONTRIBUTING.md).
`;

  fs.writeFileSync(
    path.join(DOCS, 'learn', 'index.md'),
    `---
title: Learn
description: Sequential curriculum — work through courses in order
---

# Learn

${body}`,
  );
}

function writeLearningPath(curriculum) {
  const parts = [
    { name: 'Part 1 — Understand AI', start: 0, end: 5 },
    { name: 'Part 2 — Build applications', start: 5, end: 11 },
    { name: 'Part 3 — Production', start: 11, end: 14 },
    { name: 'Part 4 — Advanced', start: 14, end: 16 },
  ];

  const lines = [
    'One sequential path · Open each course and follow its lessons in order',
    '',
    '!!! tip "Easiest way to browse"',
    '    Use the **Learn** tab in the site nav — courses are numbered 01–16 with no module codes.',
    '',
    'Full interactive list: [Learn overview](learn/index.md)',
    '',
  ];

  for (const part of parts) {
    lines.push(`## ${part.name}`, '', '| # | Course |', '|---|--------|');
    curriculum.courses.slice(part.start, part.end).forEach((course, j) => {
      const n = part.start + j + 1;
      lines.push(`| ${n} | [${course.title}](${course.path}/index.md) |`);
    });
    lines.push('');
  }

  lines.push('## Optional tracks', '');
  for (const t of curriculum.tracks) {
    lines.push(`- [${t.title}](${t.path}/index.md)`);
  }

  fs.writeFileSync(
    path.join(DOCS, 'learning-path.md'),
    `---
title: Learning Path
---

# Learning Path

${lines.join('\n')}`,
  );
}

function mkdocsBase() {
  return `site_name: AI Engineering Handbook
site_description: The free, open-source path from transformers to production AI — RAG, agents, harnesses, evals, and LLMOps.
site_url: https://psssnikhil.github.io/learn-ai-engineering/
repo_url: https://github.com/psssnikhil/learn-ai-engineering
repo_name: Star on GitHub
edit_uri: edit/main/docs/

theme:
  name: material
  logo: assets/logo.svg
  favicon: assets/favicon.svg
  icon:
    repo: fontawesome/brands/github
  font:
    text: Inter
    code: JetBrains Mono
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
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.top
    - navigation.indexes
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.action.edit

extra_css:
  - stylesheets/extra.css

extra:
  generator: false
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/psssnikhil/learn-ai-engineering
  announcement:
    text: "v1.0 — Free handbook for AI engineers. Star on GitHub if this helps you ship."
    link: https://github.com/psssnikhil/learn-ai-engineering

copyright: Copyright &copy; 2026 AI Engineering Handbook contributors · MIT License

extra_javascript:
  - javascripts/mathjax-config.js
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
  - javascripts/mermaid-init.js

plugins:
  - search

markdown_extensions:
  - admonition
  - pymdownx.arithmatex:
      generic: false
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true

validation:
  omitted_files: ignore
  absolute_links: ignore
  unrecognized_links: ignore

`;
}

export function syncNav() {
  const curriculum = loadCurriculum();
  const navYaml = buildNavYaml(curriculum);
  fs.writeFileSync(path.join(ROOT, 'mkdocs.yml'), `${mkdocsBase()}${navYaml}\n`);
  writeLearnIndex(curriculum);
  writeLearningPath(curriculum);
  console.log(`Nav synced: ${curriculum.courses.length} courses, ${curriculum.tracks.length} tracks.`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  syncNav();
}
