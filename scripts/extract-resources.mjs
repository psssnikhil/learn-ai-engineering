#!/usr/bin/env node
/**
 * Scan docs/ for arXiv, YouTube, and common doc links → resources/*.md
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(__dirname, '../docs');
const RESOURCES = path.join(__dirname, '../resources');

const ARXIV = /https?:\/\/arxiv\.org\/[^\s)\]"']+/gi;
const YOUTUBE = /https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=[\w-]+|youtu\.be\/[\w-]+)[^\s)\]"']*/gi;
const HTTP = /https?:\/\/[^\s)\]"']+/gi;

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, files);
    else if (entry.name.endsWith('.md')) files.push(p);
  }
  return files;
}

function uniqueSorted(items) {
  return [...new Set(items)].sort();
}

function categorize(url) {
  if (/arxiv\.org/i.test(url)) return 'papers';
  if (/youtube\.com|youtu\.be/i.test(url)) return 'videos';
  if (/github\.com/i.test(url)) return 'tools';
  if (/langchain|pinecone|openai|anthropic|huggingface|docs\./i.test(url)) return 'tools';
  return 'other';
}

function main() {
  const allUrls = [];
  for (const file of walk(DOCS)) {
    const text = fs.readFileSync(file, 'utf-8');
    const matches = text.match(HTTP) || [];
    for (let url of matches) {
      url = url.replace(/[.,;:]+$/, '');
      if (!url.includes('github.io/learn-ai-engineering')) allUrls.push(url);
    }
  }

  const buckets = { papers: [], videos: [], tools: [], other: [] };
  for (const url of uniqueSorted(allUrls)) {
    buckets[categorize(url)].push(url);
  }

  fs.mkdirSync(RESOURCES, { recursive: true });

  writeList(path.join(RESOURCES, 'papers.md'), 'Papers & Research', buckets.papers, 'arXiv and research links extracted from lessons.');
  writeList(path.join(RESOURCES, 'videos.md'), 'Videos', buckets.videos, 'YouTube and video links from lessons.');
  writeList(path.join(RESOURCES, 'tools-and-libraries.md'), 'Tools & Libraries', buckets.tools, 'Documentation, SDKs, and GitHub repos referenced in lessons.');
  writeList(path.join(RESOURCES, 'other-links.md'), 'Other Links', buckets.other.slice(0, 200), 'Additional references (truncated if very long).');

  // Copy into docs for MkDocs
  const docsResources = path.join(DOCS, 'resources');
  fs.mkdirSync(docsResources, { recursive: true });
  fs.copyFileSync(path.join(RESOURCES, 'papers.md'), path.join(docsResources, 'papers.md'));
  fs.copyFileSync(path.join(RESOURCES, 'videos.md'), path.join(docsResources, 'videos.md'));
  fs.copyFileSync(path.join(RESOURCES, 'tools-and-libraries.md'), path.join(docsResources, 'tools-and-libraries.md'));

  patchMkdocsResourcesNav();
  fs.writeFileSync(
    path.join(docsResources, 'index.md'),
    fs.readFileSync(path.join(RESOURCES, 'README.md'), 'utf-8'),
  );

  console.log(`Extracted: ${buckets.papers.length} papers, ${buckets.videos.length} videos, ${buckets.tools.length} tools.`);
}

function patchMkdocsResourcesNav() {
  const mkdocsPath = path.join(__dirname, '../mkdocs.yml');
  if (!fs.existsSync(mkdocsPath)) return;
  let yml = fs.readFileSync(mkdocsPath, 'utf-8');
  if (yml.includes('Tools & Libraries: resources/tools-and-libraries.md')) return;
  yml = yml.replace(
    '  - Resources: resources/index.md',
    `  - Resources:
      - Overview: resources/index.md
      - Papers: resources/papers.md
      - Videos: resources/videos.md
      - Tools & Libraries: resources/tools-and-libraries.md`,
  );
  fs.writeFileSync(mkdocsPath, yml);
}

function writeList(filePath, title, urls, intro) {
  const lines = [
    `# ${title}`,
    '',
    intro,
    '',
    `_Auto-generated from lesson content. Edit freely and re-run \`npm run extract-resources\`._`,
    '',
  ];
  for (const url of urls) {
    lines.push(`- ${url}`);
  }
  lines.push('');
  fs.writeFileSync(filePath, lines.join('\n'));
}

main();
