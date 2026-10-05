const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = (process.argv[2] || 'http://127.0.0.1:4176').replace(/\/$/, '');
const width = Number(process.argv[3] || 934);
const output = process.argv[4];
const all = process.argv.includes('--all');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width, height: 887 } });
  const errors = [], results = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(`${base}/2026/10/05/transformer-foundations/`, { waitUntil: 'domcontentloaded', timeout: 20000 });
    async function prepare(name) {
      const element = page.locator(`iframe[data-widget="${name}"]`);
      await element.scrollIntoViewIfNeeded();
      const shell = await (await element.elementHandle()).contentFrame();
      await shell.waitForSelector('#codex-visualization', { timeout: 20000 });
      const content = await (await shell.locator('#codex-visualization').elementHandle()).contentFrame();
      await content.waitForSelector('select,button,input', { timeout: 20000 });
      await content.evaluate(() => document.fonts.ready);
      return { element, shell, content };
    }
    async function stable(name, phase, frame) {
      await page.waitForTimeout(700);
      const values = [];
      for (let i = 0; i < 25; i++) {
        values.push({
          height: await frame.element.evaluate(n => n.clientHeight),
          innerHeight: await frame.shell.locator('#codex-visualization').evaluate(n => n.clientHeight),
          scroll: await page.evaluate(() => scrollY),
        });
        await page.waitForTimeout(80);
      }
      const outer = [...new Set(values.map(v => v.height))];
      const inner = [...new Set(values.map(v => v.innerHeight))];
      const scroll = [...new Set(values.map(v => v.scroll))];
      results.push({ name, phase, outer, inner, scroll });
      assert.equal(outer.length, 1, `${name}/${phase}: article frame height oscillates: ${outer}`);
      assert.equal(inner.length, 1, `${name}/${phase}: nested frame height oscillates: ${inner}`);
      assert.equal(scroll.length, 1, `${name}/${phase}: scroll position oscillates: ${scroll}`);
      const layout = await frame.shell.evaluate(() => ({
        shell: document.body.getBoundingClientRect().height,
        iframe: document.querySelector('#codex-visualization').getBoundingClientRect().height,
      }));
      assert.ok(Math.abs(outer[0] - layout.shell) <= 1, `${name}: content clipped or blank space left after resize`);
      return outer[0];
    }
    const names = all ? ['transformer-foundations','attention-shapes','norms-and-residuals','rope-rotation','mla-cache'] : ['attention-shapes','rope-rotation'];
    for (const name of names) {
      const frame = await prepare(name);
      await stable(name, 'initial', frame);
      if (name === 'attention-shapes') {
        await frame.content.locator('#as-next').click();
        assert.match(await frame.content.locator('body').innerText(), /2 \/ 10/);
        await stable(name, 'next-step', frame);
        await frame.content.locator('#as-prev').click();
        await stable(name, 'previous-step', frame);
      } else if (name === 'rope-rotation') {
        const score = text => text.match(/旋转后点积[：:]\s*([-\d.]+)/)?.[1];
        const before = score(await frame.content.locator('body').innerText());
        await frame.content.locator('#rp-shift').click();
        assert.equal(score(await frame.content.locator('body').innerText()), before);
        await stable(name, 'shift-positions', frame);
      } else if (name === 'transformer-foundations') {
        await frame.content.locator('#tf-view').selectOption('cost');
        await frame.content.getByText('单层线性权重',{exact:true}).waitFor();
        await stable(name, 'change-view', frame);
      } else if (name === 'norms-and-residuals') {
        await frame.content.locator('#nr-view').selectOption('residual');
        await stable(name, 'change-view', frame);
      } else if (name === 'mla-cache') {
        await frame.content.locator('#ml-next').click();
        await stable(name, 'next-step', frame);
      }
    }
    if (process.argv.includes('--resize')) {
      for (const nextWidth of [390, 934]) {
        await page.setViewportSize({ width: nextWidth, height: 887 });
        for (const name of ['attention-shapes', 'rope-rotation']) {
          await stable(name, `resize-${nextWidth}`, await prepare(name));
        }
      }
    }
    const metrics = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth }));
    assert.ok(metrics.document <= metrics.viewport, `Page overflow at ${width}px`);
    assert.deepEqual(errors, [], 'Browser script errors');
    console.log(JSON.stringify({ width, checkedStates: results.length, stable: true, pageErrors: errors }));
  } finally {
    if (output) fs.writeFileSync(output, JSON.stringify({ base, width, results, errors }, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
