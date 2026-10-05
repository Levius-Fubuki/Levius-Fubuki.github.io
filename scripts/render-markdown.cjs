#!/usr/bin/env node
// Render opted-in math posts with the site's existing KaTeX fonts and CSS.
const fs = require('node:fs');
const { marked } = require('marked');
const path = process.argv[2];
if (!path) throw new Error('Usage: node scripts/render-markdown.cjs source.md [--math]');
let source = fs.readFileSync(path, 'utf8');
if (!process.argv.includes('--math')) {
  process.stdout.write(marked(source));
  process.exit(0);
}
const katex = require('./vendor/katex-0.16.22.cjs');
const protectedCode = [];
source = source.replace(/```[^\n]*\n[\s\S]*?```|`[^`\n]+`/g, code => {
  const token = `CODEPROTECTEDTOKEN${protectedCode.length}END`;
  protectedCode.push(code);
  return token;
});
const formulas = [];
function formula(tex, displayMode) {
  const token = `MATHTYPESSETTOKEN${formulas.length}END`;
  const html = katex.renderToString(tex.trim(), {
    displayMode, throwOnError: true, strict: 'ignore', trust: false,
    output: 'htmlAndMathml',
  });
  formulas.push({ token, html, displayMode });
  return token;
}
source = source.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => formula(tex, true));
source = source.replace(/\$(?!\$)([^\n$]+?)\$/g, (_, tex) => formula(tex, false));
protectedCode.forEach((code, i) => {
  source = source.replace(`CODEPROTECTEDTOKEN${i}END`, code);
});
let rendered = marked(source);
formulas.forEach(({ token, html, displayMode }) => {
  if (displayMode) rendered = rendered.replace(`<p>${token}</p>`, html);
  rendered = rendered.replaceAll(token, html);
});
process.stdout.write(rendered);
