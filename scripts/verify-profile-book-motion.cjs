const {chromium} = require('playwright');
const fs = require('node:fs');
const base = 'http://127.0.0.1:4173';
const out = 'docs/design/profile-book-motion';
(async () => {
  const browser = await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
  const context = await browser.newContext({viewport:{width:924,height:889}});
  const page = await context.newPage(), checks = [], errors = [], failed = [];
  page.on('pageerror', error => errors.push(error.stack));
  page.on('response', response => { if(response.status()>=400) failed.push(response.url()); });
  const check = (name, result) => { checks.push({name,passed:!!result});console.log(`${result?'PASS':'FAIL'} ${name}`);if(!result)throw Error(name); };
  const home = async (panel='articles') => {
    await page.goto(base+'/?view=profile-book-motion#'+panel);
    await page.waitForFunction(panel=>window.LeviusPages?.getState().page===panel&&!window.LeviusPages.getState().moving,panel);
    await page.waitForTimeout(1100);
    if(panel==='articles' && await page.locator('.collection-detail').isVisible() && await page.locator('[data-book=llm-roadmap]').isHidden()){await page.locator('.collection-return').click();await page.waitForSelector('.collection-shelf:visible');}
    if(panel==='articles' && await page.locator('.collection-shelf').isVisible()){await page.locator('[data-open-collection=llm]').click();if(await page.locator('.collection-reader').isVisible()) await page.locator('.reader-all').click();await page.waitForSelector('.collection-detail:visible');await page.waitForTimeout(350);}
  };
  const cardReady = () => page.waitForFunction(()=>document.querySelector('.identity-card').contentWindow.__holo?.ready);
  const cardState = () => page.evaluate(()=>document.querySelector('.identity-card').contentWindow.__holo.getState());
  const shot = async file => {
    await page.locator('.profile-backdrop').evaluateAll(images=>Promise.all(images.map(image=>image.decode())));
    await page.waitForFunction(()=>document.querySelectorAll('[data-decoding]').length===0);
    await page.screenshot({path:out+'/'+file+'.png'});
  };
  try {
    await home('profile');await cardReady();await page.locator('.profile-backdrop').evaluate(image=>image.decode());
    check('background covers both halves of the hero',await page.locator('.profile-backdrop').evaluate(image=>{
      const a=image.getBoundingClientRect(), b=image.closest('.hero-scene').getBoundingClientRect();return Math.abs(a.left-b.left)<1&&Math.abs(a.width-b.width)<1&&Math.abs(a.height-b.height)<1&&image.naturalWidth>0;
    }));
    check('embedded card HTML and renderer are transparent',await page.evaluate(()=>{
      const w=document.querySelector('.identity-card').contentWindow;
      return w.getComputedStyle(w.document.body).backgroundColor==='rgba(0, 0, 0, 0)'&&w.__holo.renderer.getClearAlpha()===0&&w.__holo.renderer.getContext().getContextAttributes().alpha;
    }));
    check('ambient canvas spans the full scene',await page.locator('.ambient-field').evaluate(canvas=>canvas.clientWidth===canvas.parentElement.clientWidth));
    const stage=page.frameLocator('.identity-card').locator('#stage');
    await stage.click();check('original card click still flips', (await cardState()).flipped);
    await stage.click({button:'right'});check('original right-click reset still works',!(await cardState()).flipped);
    const bounds=await stage.boundingBox();await page.mouse.move(bounds.x+bounds.width*.5,bounds.y+bounds.height*.5);await page.mouse.down();await page.waitForTimeout(380);
    check('long press still starts card dragging',await stage.evaluate(node=>node.classList.contains('dragging')));
    await page.mouse.move(bounds.x+bounds.width*.5+55,bounds.y+bounds.height*.5+24,{steps:5});await page.mouse.up();
    check('card drag releases cleanly',!await stage.evaluate(node=>node.classList.contains('dragging')));
    await stage.click({button:'right'});await page.waitForTimeout(400);await shot('profile-desktop');
    await page.locator('.motion-toggle').click();check('ambient motion toggle stops scene',await page.evaluate(()=>!window.LeviusMotion.getState().enabled&&!document.documentElement.classList.contains('motion-running')));await page.locator('.motion-toggle').click();

    for(const width of [390,320]) {
      await page.setViewportSize({width,height:844});await page.waitForTimeout(400);
      check(`${width}px uses portrait composition`,await page.locator('.profile-backdrop').evaluate(image=>image.currentSrc.includes('profile-medieval-mobile.webp')));
      check(`${width}px profile has no horizontal overflow`,await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      check(`${width}px profile actions stay inside scene`,await page.locator('.hero-actions').evaluate(node=>{const a=node.getBoundingClientRect(),b=node.closest('.hero-scene').getBoundingClientRect();return a.left>=b.left&&a.right<=b.right&&a.bottom<=b.bottom;}));
      await shot('profile-'+width);
    }
    await page.setViewportSize({width:924,height:889});
    await home();
    const cover=page.locator('.book-link').first(),href=await cover.getAttribute('href');
    const started=Date.now();await cover.click();
    await page.waitForSelector('.book-opening[open]');
    check('cover click opens modal book scene',await page.locator('.book-opening').count()===1);
    check('opening uses the clicked cover',await page.locator('.book-opening .book-art').getAttribute('src')==='/img/books/llm-roadmap-color.webp');
    check('book has two separately turning sheets',await page.locator('.book-turning-leaf').count()===2);
    await page.waitForTimeout(450);await page.screenshot({path:out+'/book-opening-desktop.png'});
    check('cover actually rotates around spine',await page.locator('.book-turning-cover').evaluate(node=>getComputedStyle(node).transform.startsWith('matrix3d')));
    await page.waitForURL(url=>url.pathname===new URL(href,base).pathname);
    const navigationTime=Date.now()-started;
    await page.waitForFunction(()=>document.querySelector('.inner-title')?.textContent==='Becoming an LLM Engineer: A Learning Roadmap from Scratch');
    check('animation leads to the selected article',await page.locator('.inner-title').innerText()==='Becoming an LLM Engineer: A Learning Roadmap from Scratch');
    check('navigation waits for page turn',navigationTime>=900);
    await page.goBack();await page.waitForFunction(()=>window.LeviusPages?.getState().page==='articles');
    check('Back removes transition and unlocks page',await page.locator('.book-opening').count()===0&&!await page.locator('html').evaluate(node=>node.classList.contains('book-is-opening')));
    check('Back restores the selected collection',await page.locator('.collection-detail').isVisible()&&await page.locator('.shelf-book:not([hidden])').count()===2);await page.locator('.collection-return').click();await page.waitForSelector('.collection-shelf:visible');await page.locator('[data-open-collection=rl]').click();if(await page.locator('.collection-reader').isVisible()) await page.locator('.reader-all').click();await page.waitForSelector('.collection-detail:visible');check('shelf usable after Back',await page.locator('.shelf-book:not([hidden])').count()===5);

    await home();await page.locator('.book-link').first().click();await page.waitForSelector('.book-opening[open]');await page.keyboard.press('Escape');await page.waitForURL('**/2026/04/07/**');
    check('Escape skips animation and completes navigation',await page.locator('.inner-title').count()===1);
    await home();await page.locator('.book-link').first().click();await page.waitForSelector('.book-opening[open]');
    await page.locator('.book-opening').evaluate(node=>node.getAnimations({subtree:true}).forEach(animation=>animation.cancel()));await page.waitForURL('**/2026/04/07/**');
    check('cancelled animation cannot strand navigation',await page.locator('.inner-title').count()===1);

    await page.emulateMedia({reducedMotion:'reduce'});await home();
    await page.evaluate(()=>{window.__openingSeen=false;new MutationObserver(records=>{for(const record of records)for(const node of record.addedNodes)if(node.classList?.contains('book-opening'))window.__openingSeen=true;}).observe(document.body,{childList:true});});
    const navigation=page.waitForURL('**/2026/04/07/**'),directStart=Date.now();await page.locator('.book-link').first().click();await navigation;
    check('reduced motion navigates directly',Date.now()-directStart<900&&await page.locator('.inner-title').count()===1&&await page.locator('.book-opening').count()===0);
    await page.emulateMedia({reducedMotion:'no-preference'});
    await home();await page.evaluate(()=>localStorage.setItem('levius-ambient-motion','off'));await page.goto(base+'/archives/');
    await page.locator('.book-preview-toggle').first().click();await page.locator('.archive-cover-link').first().click();await page.waitForURL('**/2026/04/07/**');
    check('saved motion-off preference applies to archive covers',await page.locator('.book-opening').count()===0);
    await page.evaluate(()=>localStorage.removeItem('levius-ambient-motion'));

    await home();
    const popupPromise=context.waitForEvent('page');await page.locator('.book-link').first().click({modifiers:['Meta']});const popup=await popupPromise;await popup.waitForLoadState();
    check('modified click retains new-tab navigation',await page.locator('.book-opening').count()===0&&new URL(popup.url()).pathname.startsWith('/2026/04/07/'));await popup.close();

    await page.setViewportSize({width:390,height:844});await home();await page.locator('.book-link').first().click();await page.waitForSelector('.book-opening[open]');await page.waitForTimeout(400);await page.screenshot({path:out+'/book-opening-mobile.png'});await page.waitForURL('**/2026/04/07/**');
    check('mobile page turn reaches article',await page.locator('.inner-title').count()===1);
    await page.goto(base+'/card/');await page.waitForFunction(()=>window.__holo?.ready);
    check('standalone card keeps its original solid background',await page.evaluate(()=>window.__holo.renderer.getClearAlpha()===1&&getComputedStyle(document.body).backgroundColor==='rgb(16, 16, 16)'));
    check('no JavaScript errors',errors.length===0);check('no missing resources',failed.length===0);
  } finally {fs.writeFileSync(out+'/report.json',JSON.stringify({checks,errors,failed},null,2));await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
