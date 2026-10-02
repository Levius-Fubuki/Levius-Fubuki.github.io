const {chromium}=require('playwright');
const fs=require('node:fs');
(async()=>{
 const b=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 const base=process.env.CARD_TEST_BASE||'http://127.0.0.1:4173';
 const page=async()=>{const p=await b.newPage({viewport:{width:600,height:550}});
  await p.route('https://raw.githubusercontent.com/Levius-Fubuki/Levius-Fubuki.github.io/main/card/**',async route=>{
   const path=new URL(route.request().url()).pathname.split('/main/')[1];
   if(!fs.existsSync(path))throw Error('Missing mirror fixture '+path);
   await route.fulfill({path,headers:{'access-control-allow-origin':'*'}});
  });return p;};
 const check=(n,ok)=>{if(!ok)throw Error(n);console.log('PASS '+n)};
 try{
  const p=await page();const errors=[];p.on('pageerror',e=>errors.push(e.message));await p.goto(base+'/card/?embed=1');await p.waitForFunction(()=>window.__holo?.ready);
  check('normal card initializes without errors',!errors.length);check('card has no external render-blocking stylesheets',await p.locator('link[rel=stylesheet]').count()===0);
  check('one bundled module entry',await p.locator('script[type=module]').count()===1);
  await p.locator('#stage').click();check('card still flips',(await p.evaluate(()=>window.__holo.getState())).flipped);await p.locator('#stage').click({button:'right'});check('reset still works',!(await p.evaluate(()=>window.__holo.getState())).flipped);await p.close();
  const q=await page();let attempts=0;await q.route('**/assets/subject.*.webp*',route=>++attempts===1?route.abort('timedout'):route.fallback());await q.goto(base+'/card/?embed=1');await q.waitForFunction(()=>window.__holo?.ready);check('a failed texture retries and recovers automatically',attempts===2);await q.close();
  const slow=await page();let starts=0;await slow.route('**/assets/subject.*.webp*',async route=>{if(++starts===1){await new Promise(r=>setTimeout(r,7000));await route.abort().catch(()=>{});}else await route.fallback()});await slow.goto(base+'/card/?embed=1',{waitUntil:'domcontentloaded'});await slow.locator('.card-loading-preview').evaluate(i=>i.decode());check('real preview stays visible while a request stalls',await slow.locator('.card-loading-preview').isVisible());check('loading surface preserves the profile background',await slow.locator('#loading').evaluate(n=>getComputedStyle(n).backgroundColor==='rgba(0, 0, 0, 0)'));await slow.waitForFunction(()=>window.__holo?.ready,{},{timeout:30000});check('hung request is bounded and second attempt recovers',starts===2);await slow.close();
  const fail=await page();let failures=0;await fail.route('**/assets/subject.*.webp*',route=>{failures++;return route.abort('timedout')});await fail.goto(base+'/card/?embed=1');await fail.waitForFunction(()=>window.__holo?.error);await fail.locator('.card-loading-preview').evaluate(i=>i.decode());check('permanent failure stops after three attempts',failures===3);check('permanent failure retains artwork and a useful error',await fail.locator('.card-loading-preview').isVisible()&&(await fail.locator('[data-card-status]').innerText()).includes('subject.'));await fail.unroute('**/assets/subject.*.webp*');await fail.locator('[data-card-retry]').click();await fail.waitForFunction(()=>window.__holo?.ready);check('retry recovers without stale loader state',await fail.locator('#loading').count()===0);await fail.close();
  const blocked=await page();await blocked.route('**/app.bundle.*.js*',r=>r.abort());await blocked.goto(base+'/card/?embed=1');await blocked.locator('.card-loading-preview').evaluate(i=>i.decode());check('preview works even when the JS bundle fails',await blocked.locator('.card-loading-preview').isVisible());fs.mkdirSync('docs/design/card-loading',{recursive:true});await blocked.screenshot({path:'docs/design/card-loading/module-blocked.png'});await blocked.close();
  const mirror=await page();let mirrorLoads=0;mirror.on('response',r=>{if(r.url().startsWith('https://raw.githubusercontent.com/'))mirrorLoads++});
  await mirror.route(base+'/card/assets/**',r=>r.abort('timedout'));
  await mirror.route(base+'/card/app.bundle.*.js*',r=>r.abort('timedout'));
  await mirror.goto(base+'/card/?embed=1');await mirror.waitForFunction(()=>window.__holo?.ready);
  check('real WebGL card initializes when every local runtime and texture request fails',mirrorLoads>=13&&await mirror.evaluate(()=>window.__holo.renderer.isWebGLRenderer&&window.__cardRuntimeSource==='https://raw.githubusercontent.com'));
  await mirror.locator('#stage').click();check('mirror-loaded card flips interactively',await mirror.evaluate(()=>window.__holo.getState().flipped));await mirror.close();
 }finally{await b.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
