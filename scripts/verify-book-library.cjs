const {chromium} = require('playwright');
const fs = require('node:fs');
const data = JSON.parse(fs.readFileSync('data/book-library.en.json', 'utf8'));
const base = 'http://127.0.0.1:4173';
const out = 'docs/design/book-library';

(async () => {
  const browser = await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:true});
  const context = await browser.newContext({viewport:{width:924,height:889}, reducedMotion:'reduce'});
  const page = await context.newPage();
  const checks = [], errors = [], failed = [];
  const check = (name, passed) => {
    checks.push({name, passed:!!passed});
    console.log(`${passed ? 'PASS' : 'FAIL'} ${name}`);
    if (!passed) throw Error(name);
  };
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if(response.status() >= 400) failed.push(response.url()); });
  const ready = async () => { await page.evaluate(() => document.fonts.ready); };
  const screenshot = async options => {
    // Wait for async image decoding as well as network completion before visual QA.
    await page.locator('.book-art').evaluateAll(images => Promise.all(images.map(image => {
      image.loading = 'eager';
      return image.decode();
    })));
    await page.screenshot(options);
  };
  const panel = async name => {
    await page.waitForFunction(name => window.LeviusPages?.getState().page === name && !window.LeviusPages.getState().moving, name);
  };
  try {
    await page.goto(base + '/?view=book-library#welcome'); await panel('welcome'); await ready();
    check('exact opening name', (await page.locator('#opening-title').innerText()).replace(/\s+/g,' ').trim() === "Levius_Fubuki's BLOGS");
    await screenshot({path:out+'/welcome-desktop.png'});
    await page.locator('.page-dots button').nth(3).click(); await panel('articles');
    check('all eight volumes in homepage shelf', await page.locator('.shelf-book').count() === 8);
    check('homepage remains four panels', await page.locator('.page-panel').count() === 4);
    check('initial view shows only collection spines',await page.locator('.collection-spine:visible').count()===3 && !await page.locator('.book-rail').isVisible());
    await page.screenshot({path:out+'/cabinet-desktop.png'});
    await page.locator('[data-open-collection="rl"]').click();
    if(await page.locator('.collection-reader').isVisible()) await page.locator('.reader-all').click();await page.waitForSelector('.collection-detail:visible');
    check('shelf next enabled at start', await page.locator('[data-shelf-step="1"]').isEnabled());
    await page.locator('[data-shelf-step="1"]').click();
    await page.waitForFunction(() => document.querySelector('.book-rail').scrollLeft > 100);
    check('shelf arrows scroll within article panel', await page.evaluate(() => document.body.dataset.page === 'articles'));
    const choose=async id=>{await page.locator('.collection-return').click();await page.waitForSelector('.collection-shelf:visible');await page.locator(`[data-open-collection="${id}"]`).click();if(await page.locator('.collection-reader').isVisible()) await page.locator('.reader-all').click();await page.waitForSelector('.collection-detail:visible');};
    await choose('llm');
    check('large model collection exposes two books', await page.locator('.shelf-book:not([hidden])').count() === 2);
    await choose('rl');
    check('reinforcement learning collection exposes five books', await page.locator('.shelf-book:not([hidden])').count() === 5);
    await choose('notes');
    check('prologue collection exposes original first post', await page.locator('.shelf-book:not([hidden])').getAttribute('data-book') === 'hello-world');
    await choose('rl');
    await page.locator('.book-rail').focus(); await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('.book-rail').scrollLeft > 100);
    check('keyboard scrolls shelf', await page.evaluate(() => document.body.dataset.page === 'articles'));
    await choose('llm');
    await screenshot({path:out+'/shelf-desktop.png'});

    for (const book of data.books) {
      await page.goto(base + book.path); await ready();
      const art = page.locator('.article-book .book-art');
      await art.scrollIntoViewIfNeeded();
      await page.waitForFunction(() => { const img=document.querySelector('.article-book .book-art');return img?.complete&&img.naturalWidth>0; });
      check('matching article cover: ' + book.id, (await art.getAttribute('src')) === book.art);
      check('article text remains available: ' + book.id, (await page.locator('#article-container').innerText()).length > 100);
      if(book.id==='llm-roadmap') await screenshot({path:out+'/article-cover-desktop.png'});
    }

    await page.goto(base + '/archives/'); await ready();
    const row = page.locator('.has-book-preview').first();
    check('eight archive previews available', await page.locator('.has-book-preview').count() === 8);
    check('archive starts collapsed', await row.locator('.archive-book-preview').evaluate(node => node.inert && node.getBoundingClientRect().height < 2));
    await row.hover();
    check('hover expands archive row', await row.locator('.book-preview-toggle').getAttribute('aria-expanded') === 'true');
    check('expanded preview includes content summary', await row.locator('.archive-book-copy').innerText().then(text=>text.includes(data.books[0].summary)));
    await screenshot({path:out+'/archive-expanded.png'});
    await page.mouse.move(2,2);
    check('pointer leaving closes unpinned preview', await row.locator('.book-preview-toggle').getAttribute('aria-expanded') === 'false');
    await row.locator('.article-sort-item-title').focus();
    check('keyboard focus opens archive preview', await row.locator('.book-preview-toggle').getAttribute('aria-expanded') === 'true');
    await page.keyboard.press('Escape');
    check('Escape closes archive preview', await row.locator('.book-preview-toggle').getAttribute('aria-expanded') === 'false');
    await row.locator('.book-preview-toggle').click(); await page.mouse.move(2,2);
    check('disclosure pins archive preview', await row.locator('.book-preview-toggle').getAttribute('aria-expanded') === 'true');
    await row.locator('.book-preview-toggle').click();
    check('disclosure closes archive preview', await row.locator('.book-preview-toggle').getAttribute('aria-expanded') === 'false');

    await page.goto(base + '/categories/'); await ready();
    check('two parent collections', await page.locator('.collection-set').count() === 2);
    const branch = page.locator('.collection-branch');
    check('deep learning stays a large-model branch', await branch.evaluate(node => node.closest('[data-collection]').dataset.collection === 'llm'));
    await screenshot({path:out+'/collections-desktop.png',fullPage:true});
    await branch.click();
    check('branch routes to deep-learning volume', await page.locator('.has-book-preview').count() === 1 && (await page.locator('.article-sort-item-title').innerText()) === 'Deep Learning Fundamentals');

    for(const width of [390,320]) {
      await page.setViewportSize({width,height:844});
      for(const [name,path] of [['welcome','/?view=book-library#welcome'],['shelf','/?view=book-library#articles'],['collections','/categories/'],['archive','/archives/'],['article',data.books[0].path]]) {
        await page.goto(base+path); await ready();
        if(name==='welcome') await panel('welcome');
        if(name==='shelf') {await panel('articles');check(`${width}px only collection spines initially`,await page.locator('.collection-spine:visible').count()===3&&!await page.locator('.book-rail').isVisible());await page.screenshot({path:out+`/cabinet-${width}.png`});await page.locator('[data-open-collection=rl]').click();if(await page.locator('.collection-reader').isVisible()) await page.locator('.reader-all').click();await page.waitForSelector('.collection-detail:visible');}
        if(name==='archive') await page.locator('.book-preview-toggle').first().click();
        check(`${width}px ${name} fits`, await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
        if(name==='welcome') check(`${width}px opening title visible`, await page.locator('#opening-title').evaluate(node=>{const r=node.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth;}));
        if(name==='shelf') check(`${width}px shelf has horizontal overflow`, await page.locator('.book-rail').evaluate(node=>node.scrollWidth>node.clientWidth));
        if(name==='article') check(`${width}px cover and contents available`, await page.locator('.article-book').count()===1&&await page.locator('.contents-panel').count()===1);
        await screenshot({path:out+`/${name}-${width}.png`});
      }
    }
    const touchContext = await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,reducedMotion:'reduce'});
    const touch = await touchContext.newPage();
    await touch.goto(base+'/archives/');
    const toggle = touch.locator('.book-preview-toggle').first();
    await toggle.tap();
    check('touch opens archive preview',await toggle.getAttribute('aria-expanded')==='true');
    await toggle.tap();
    check('touch closes archive preview',await toggle.getAttribute('aria-expanded')==='false');
    await touchContext.close();

    await page.setViewportSize({width:924,height:889});
    await page.emulateMedia({reducedMotion:'no-preference'});
    await page.goto(base+'/?view=book-library#welcome');await panel('welcome');
    await page.waitForTimeout(1200);
    check('decoding restores new opening name', (await page.locator('#opening-title').innerText()).replace(/\s+/g,' ').trim()==="Levius_Fubuki's BLOGS");
    check('no missing page resources',failed.length===0);
    check('no JavaScript errors',errors.length===0);
  } finally {
    fs.writeFileSync(out+'/report.json',JSON.stringify({checks,errors,failed},null,2));
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
