const {chromium}=require('playwright');const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',permissions:['clipboard-read','clipboard-write']});
 const page=await context.newPage(),checks=[],errors=[],failed=[];const out='docs/design/inner-pages';
 const manifest=JSON.parse(fs.readFileSync(out+'/manifest.json'));page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)failed.push(r.url())});
 const check=(name,value)=>{checks.push({name,passed:!!value});console.log(`${value?'PASS':'FAIL'} ${name}`);if(!value)throw Error(name)};
 const url=path=>'http://127.0.0.1:4173/'+path.replace(/index.html$/,'');
 const shots={'archives/index.html':'archive','categories/index.html':'categories','tags/index.html':'tags','Gallery/index.html':'gallery','2026/04/06/4-Actor-Critic/index.html':'article','categories/强化学习/index.html':'category-detail','tags/DQN/index.html':'tag-detail'};
 try{
 check('all 33 original pages included',manifest.pages.length===33);
 for(const entry of manifest.pages){
  await page.goto(url(entry.path),{waitUntil:'load'});await page.evaluate(()=>document.fonts.ready);
  check('desktop shell '+entry.path,await page.locator('.inner-page .site-header').isVisible()&&await page.locator('.inner-title').innerText()===entry.title);
  if(entry.kind==='post')check('math and images preserved '+entry.title,await page.locator('#article-container .katex').count()===entry.math_count&&await page.locator('#article-container img').count()===entry.images);
  if(shots[entry.path])await page.screenshot({path:out+'/'+shots[entry.path]+'-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  check('mobile fits '+entry.path,await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(shots[entry.path])await page.screenshot({path:out+'/'+shots[entry.path]+'-mobile.png'});
  await page.setViewportSize({width:1440,height:1000});
 }
 await page.goto('http://127.0.0.1:4173/404.html');check('branded 404',await page.locator('.inner-title').innerText()==='404 / 页面不存在');
 await page.goto(url('2026/04/06/4-Actor-Critic/index.html'));
 await page.locator('.search-open').click();await page.locator('#search-input').fill('Actor-Critic');await page.waitForFunction(()=>document.querySelectorAll('.search-result').length>0);check('shared search works',await page.locator('.search-result').count()>0);await page.keyboard.press('Escape');
 const toc=page.locator('.toc-link').nth(3),hash=await toc.getAttribute('href');await toc.click();check('TOC reaches original anchor',await page.evaluate(hash=>document.getElementById(decodeURIComponent(hash.slice(1))).getBoundingClientRect().top>=0&&document.getElementById(decodeURIComponent(hash.slice(1))).getBoundingClientRect().top<100,hash));
 await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));await page.locator('.reading-width').click();check('focus reading mode',await page.locator('body').evaluate(n=>n.classList.contains('reading-wide')));await page.locator('.reading-width').click();
 await page.locator('.copy-page').click();check('copy canonical article URL',(await page.evaluate(()=>navigator.clipboard.readText())).includes('/4-Actor-Critic/'));
 await page.goto(url('2025/03/30/hello-world/index.html'));await page.locator('.code-copy').first().click();check('copy code excludes line numbers',(await page.evaluate(()=>navigator.clipboard.readText())).trim()==='$ hexo new "My New Post"');
 await page.goto(url('2026/04/06/5-Monte Carlo Tree Search/index.html'));const img=page.locator('#article-container img').first();await img.click();check('article image opens preview',await page.locator('.image-dialog').evaluate(n=>n.open));await page.keyboard.press('Escape');check('image preview closes',!await page.locator('.image-dialog').evaluate(n=>n.open));
 await page.setViewportSize({width:320,height:740});await page.goto(url('2026/04/06/01-深度学习基础/index.html'));check('320px long article fits',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));check('mobile TOC initially collapsed',!await page.locator('.contents-panel').evaluate(n=>n.open));await page.locator('.contents-panel summary').click();check('mobile TOC opens',await page.locator('.contents-panel').evaluate(n=>n.open));await page.locator('.menu-toggle').click();check('mobile navigation opens',await page.locator('#main-nav').isVisible());
 await page.setViewportSize({width:1440,height:1000});await page.emulateMedia({reducedMotion:'no-preference'});await page.goto(url('archives/index.html'));await page.waitForTimeout(1000);check('decode restores archive title',await page.locator('.inner-title').innerText()==='文章归档');
 check('no failed resource responses',failed.length===0);check('no JavaScript errors',errors.length===0);
 }finally{fs.writeFileSync(out+'/report.json',JSON.stringify({checks,errors,failed},null,2));await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
