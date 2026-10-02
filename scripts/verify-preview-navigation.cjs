const{chromium}=require('playwright');const fs=require('node:fs');
(async()=>{const b=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});const checks=[],errors=[];const out='docs/design/preview-navigation';fs.mkdirSync(out,{recursive:true});const check=(name,ok)=>{checks.push({name,passed:!!ok});console.log((ok?'PASS ':'FAIL ')+name);if(!ok)throw Error(name)};try{const p=await b.newPage({viewport:{width:953,height:889},reducedMotion:'reduce'});p.on('pageerror',e=>errors.push(e.message));
for(const [name,url] of [['profile','http://127.0.0.1:4173/?refresh=archives-fix#profile'],['article','http://127.0.0.1:4173/2026/04/06/4-Actor-Critic/?refresh=archives-fix']]){
 const response=await p.goto(url);check(name+' HTML disables cache',response.headers()['cache-control'].includes('no-store'));
 const archive=p.locator('.main-nav').getByRole('link',{name:'归档',exact:true});const target=await archive.getAttribute('href');check(name+' archive link has preview revision',new URL(target).searchParams.has('__preview'));
 const doc=p.waitForResponse(r=>new URL(r.url()).pathname==='/archives/'&&r.request().resourceType()==='document');await archive.click();const loaded=await doc;
 check(name+' archive response disables cache',loaded.headers()['cache-control'].includes('no-store'));
 await p.waitForSelector('.inner-title');check(name+' opens redesigned archive',await p.locator('.inner-title').innerText()==='文章归档'&&await p.locator('body').evaluate(n=>n.classList.contains('page-archives')));
 check(name+' no old theme assets',await p.locator('link[href*="vaporwave"],script[src*="vaporwave"],link[href="/css/index.css"]').count()===0);
 check(name+' CSS carries preview revision',await p.locator('link[rel=stylesheet]').evaluateAll(ns=>ns.every(n=>new URL(n.href).searchParams.has('__preview'))));
 await p.screenshot({path:out+'/'+name+'-to-archive.png'});
}
await p.locator('.search-open').click();await p.locator('#search-input').fill('Actor-Critic');await p.waitForSelector('a.search-result');check('dynamic search links carry revision',new URL(await p.locator('a.search-result').first().getAttribute('href')).searchParams.has('__preview'));
const r=await p.request.get('http://127.0.0.1:4173/archives/',{headers:{'If-Modified-Since':'Thu, 01 Oct 2099 00:00:00 GMT'}});check('stale validators do not return 304',r.status()===200&&(await r.text()).includes('inner-pages.css'));
check('no JavaScript errors',errors.length===0);
}finally{fs.writeFileSync(out+'/report.json',JSON.stringify({checks,errors},null,2));await b.close()}})().catch(e=>{console.error(e);process.exitCode=1});
