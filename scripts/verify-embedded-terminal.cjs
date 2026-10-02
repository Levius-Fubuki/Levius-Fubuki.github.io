const {chromium}=require('playwright');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),checks=[],errors=[];
 const out='docs/design/terminal-embedded';fs.mkdirSync(out,{recursive:true});
 page.on('pageerror',e=>errors.push(e.message));
 const check=(name,ok)=>{checks.push({name,passed:!!ok});console.log(`${ok?'PASS':'FAIL'} ${name}`);if(!ok)throw Error(name)};
 const settle=state=>page.waitForFunction(state=>document.querySelector('#terminal').dataset.scene===state,state);
 const run=async command=>{await page.locator('#terminal-input').fill(command);await page.locator('#terminal-input').press('Enter');await page.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');};
 try {
 await page.goto('http://127.0.0.1:4173/',{waitUntil:'networkidle'});
 const backup=fs.readFileSync('docs/design/encom-backup/index.html','utf8');
 const preserved=await page.evaluate(async source=>{const original=new DOMParser().parseFromString(source,'text/html');const restored=new DOMParser().parseFromString(await (await fetch(location.href)).text(),'text/html');return ['.site-header','.hero','#articles','.site-footer','#search-dialog'].map(selector=>{
 // Enhancement scripts may toggle visibility, labels and motion attributes.
 const clean=node=>{const c=node.cloneNode(true);c.querySelectorAll('*').forEach(n=>{['hidden','aria-pressed','aria-expanded','style'].forEach(a=>n.removeAttribute(a));if(n.matches('.motion-toggle'))n.textContent='motion'});return c.outerHTML};
 return [selector,clean(original.querySelector(selector))===clean(restored.querySelector(selector))]})},backup);
 preserved.forEach(([selector,ok])=>check('original homepage preserved '+selector,ok));
 check('single original identity card',await page.locator('.identity-card').count()===1);
 check('no full-page boardroom',await page.locator('#boardroom').count()===0);
 await page.waitForFunction(()=>document.querySelector('#hero-title .text-ink').textContent==='LEVIUS');await page.screenshot({path:out+'/homepage.png'});
 await page.locator('.terminal-launch').click();await settle('terminal');await page.locator('#terminal').scrollIntoViewIfNeeded();await page.waitForTimeout(4500);await settle('terminal');
 await page.locator('#terminal').screenshot({path:out+'/terminal.png'});
 check('terminal is within homepage main',await page.locator('.site-shell main #terminal').count()===1);
 check('frames and keyboard retained',await page.locator('.table-inside').isVisible()&&await page.locator('#terminal-keyboard').isVisible());
 check('terminal palette is monochrome',await page.locator('#terminal').evaluate(root=>[root,...root.querySelectorAll('*')].every(n=>['color','backgroundColor','borderTopColor'].every(p=>{const m=getComputedStyle(n)[p].match(/^rgba?\((\d+), (\d+), (\d+)/);return !m||(m[1]===m[2]&&m[2]===m[3])}))));
 await run('search 强化学习');check('real Chinese search',(await page.locator('#terminal-output').innerText()).includes('Reinforcement Learning Fundamentals'));
 await run('cat 5');check('real article preview',(await page.locator('#terminal-output').innerText()).includes('Actor-Critic'));
 await page.locator('#terminal-input').fill('');for(const key of ['h','e','l','p'])await page.locator(`[data-terminal-key="${key}"]`).click();check('virtual keyboard input',await page.locator('#terminal-input').inputValue()==='help');
 await page.locator('[data-terminal-key="Enter"]').click();await page.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');
 await page.locator('#terminal-input').fill('run articles.exe');await page.locator('#terminal-input').press('Enter');await settle('transition');
 check('transition stays inside terminal',await page.locator('#program-transition').evaluate(n=>getComputedStyle(n).position==='absolute'&&n.closest('#terminal')!==null));
 check('hero and articles remain displayed during transition',await page.locator('.hero').isVisible()&&await page.locator('#articles').isVisible());
 await page.waitForTimeout(750);await page.locator('#terminal').screenshot({path:out+'/transition.png'});await settle('program');
 check('eight real articles in local program',await page.locator('.program-article').count()===8);
 check('run does not change homepage URL',new URL(page.url()).hash==='#terminal');
 await page.locator('#terminal').screenshot({path:out+'/articles.png'});
 await page.emulateMedia({reducedMotion:'reduce'});await page.locator('.return-terminal').click();await settle('terminal');
 await page.locator('[data-launch="reinforcement"]').click();await settle('program');check('folder launch filters five articles',await page.locator('.program-article').count()===5);
 await page.locator('.program-article a').first().click();await page.waitForURL(url=>url.pathname!=='/');check('terminal link opens article',new URL(page.url()).pathname!=='/');await page.goBack({waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.querySelector('.hero'));check('article back restores original homepage',await page.locator('.hero').isVisible()&&await page.locator('#articles').isVisible());
 await page.locator('.search-open').click();await page.locator('#search-input').fill('Actor-Critic');await page.waitForFunction(()=>document.querySelectorAll('.search-result').length>0);check('original site search works',await page.locator('.search-result').count()>0);await page.keyboard.press('Escape');
 for(const width of [390,320]){
 await page.setViewportSize({width,height:844});
 if(await page.locator('.return-terminal').isVisible()){await page.locator('.return-terminal').click();await settle('terminal')}
 await page.locator('#terminal').scrollIntoViewIfNeeded();await settle('terminal');await run('help');
 check(width+'px page fits',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 check(width+'px input usable',await page.locator('#terminal-input').isEditable());
 check(width+'px terminal contents are not clipped',await page.locator('#terminal').evaluate(root=>{const outer=root.querySelector('.terminal-stage').getBoundingClientRect();return [...root.querySelectorAll('.folder-row img,.lt-readme')].every(n=>{const r=n.getBoundingClientRect();return r.right<=outer.right&&r.bottom<=outer.bottom})}));
 await page.locator('#terminal').screenshot({path:out+'/terminal-'+width+'.png'});
 await run('run articles.exe');await settle('program');check(width+'px program fits',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 }
 check('no JavaScript errors',errors.length===0);
 }finally{fs.writeFileSync(out+'/report.json',JSON.stringify({checks,errors},null,2));await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
