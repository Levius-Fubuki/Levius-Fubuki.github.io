/* NODE_PATH=<directory containing playwright> node scripts/verify-home-terminal.cjs [base URL] */
const { chromium } = require('playwright');
const { existsSync, mkdirSync, writeFileSync } = require('node:fs');
const path = require('node:path');
(async () => {
  const executablePath = process.env.CHROME_PATH || (existsSync('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome') ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : undefined);
  const browser = await chromium.launch({ headless:true, executablePath });
  const out = path.resolve('docs/design/terminal'); mkdirSync(out,{recursive:true});
  const checks = [], errors = [];
  const check = (name, ok) => { checks.push({name,passed:!!ok}); console.log((ok ? 'PASS ' : 'FAIL ')+name); if(!ok) throw new Error(name); };
  const base = process.argv[2] || 'http://127.0.0.1:4173/';
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base,{waitUntil:'networkidle'});
    check('interactive terminal exists',await page.locator('#terminal-input').count()===1);
    const input=page.locator('#terminal-input');
    const run=async command=>{await input.fill(command);await input.press('Enter');await page.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');return command.trim().toLowerCase()==='clear' ? '' : page.locator('.terminal-response').last().innerText();};
    check('help documents commands',(await run('help')).includes('search'));
    check('root directory exists',(await run('pwd')).trim()==='/');
    check('root listing includes articles',(await run('ls')).includes('articles/'));
    check('change directory',(await run('cd articles')).includes('/articles'));
    check('all eight articles listed',(await run('ls')).includes('Hello World'));
    await page.locator('.terminal-shortcuts [data-terminal-command="ls /articles"]').click();await page.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');check('article shortcut works from article directory',(await page.locator('.terminal-response').last().innerText()).includes('Hello World'));
    check('Chinese search returns real content',(await run('search 强化学习')).includes('强化学习基本概念'));
    check('search IDs remain canonical',(await run('search "Actor-Critic"')).includes('05'));
    check('cat reads article',(await run('cat 5')).includes('Actor-Critic'));
    check('invalid article is explained',(await run('open 999')).includes('未找到'));
    check('unclosed quotes are explained',(await run('search \"unfinished')).includes('引号'));
    check('empty search is explained',(await run('search')).includes('search <关键词>'));
    check('unknown command has guidance',(await run('sudo rm -rf /')).includes('help'));
    await run('search <img src=x onerror=alert(1)>');
    check('input rendered as text',await page.locator('#terminal-output img').count()===0);
    await run('pwd');await input.fill('draft');await input.press('ArrowUp');check('history previous',await input.inputValue()==='pwd');await input.press('ArrowDown');check('history restores draft',await input.inputValue()==='draft');
    await input.fill('se');await input.press('Tab');check('Tab completion',await input.inputValue()==='search ');
    await input.fill('search /');check('slash does not open search dialog',!await page.locator('#search-dialog').evaluate(e=>e.open));
    await input.press('Control+k');check('terminal shortcuts do not open site search',!await page.locator('#search-dialog').evaluate(e=>e.open));
    await run('motion off');check('motion off operates homepage',await page.locator('.motion-toggle').getAttribute('aria-pressed')==='false');await run('motion on');
    await page.locator('.terminal-keyboard-toggle').click();await input.fill('');await page.locator('[data-terminal-key="h"]').click();await page.locator('[data-terminal-key="e"]').click();await page.locator('[data-terminal-key="l"]').click();await page.locator('[data-terminal-key="p"]').click();check('screen keyboard enters text',await input.inputValue()==='help');await page.locator('[data-terminal-key="Enter"]').click();await page.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');
    check('screen keyboard submits command',(await page.locator('.terminal-response').last().innerText()).includes('search'));
    await page.locator('.terminal-keyboard-toggle').click();await run('clear');await run('ls');await page.locator('#terminal').scrollIntoViewIfNeeded();await page.waitForTimeout(800);await page.screenshot({path:out+'/desktop.png'});
    await input.fill('open 5');await input.press('Enter');await page.waitForURL('**/4-Actor-Critic/',{waitUntil:'domcontentloaded'});check('open navigates to actual article',page.url().includes('/4-Actor-Critic/'));
    await page.goto(base,{waitUntil:'networkidle'});
    for(const width of [390,320]){await page.setViewportSize({width,height:844});await page.locator('#terminal').scrollIntoViewIfNeeded();check(width+'px no overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await run('search 强化学习');const ib=await input.boundingBox(),bb=await page.locator('#terminal-form button').boundingBox();check(width+'px submit shares input row',Math.abs(ib.y-bb.y)<10);await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));await page.waitForTimeout(800);await page.screenshot({path:out+'/mobile-'+width+'.png',fullPage:true});}
    await page.emulateMedia({reducedMotion:'reduce'});check('reduced motion command explains state',(await run('motion on')).includes('系统'));
    await run('CLEAR');check('uppercase clear removes previous output',await page.locator('#terminal-output').innerText()==='');
    const failure=await browser.newPage();await failure.route('**/js/search-index.json',route=>route.fulfill({status:503,body:'unavailable'}));await failure.goto(base,{waitUntil:'networkidle'});await failure.locator('#terminal-input').fill('search test');await failure.locator('#terminal-input').press('Enter');await failure.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');check('data error is visible',(await failure.locator('.terminal-response').last().innerText()).includes('重试'));await failure.unroute('**/js/search-index.json');await failure.locator('#terminal-input').fill('ls articles');await failure.locator('#terminal-input').press('Enter');await failure.waitForFunction(()=>document.querySelector('#terminal-form').dataset.busy==='false');check('failed data fetch retries',(await failure.locator('.terminal-response').last().innerText()).includes('Hello World'));await failure.close();
    check('no page errors',errors.length===0);
  } finally { writeFileSync(out+'/report.json',JSON.stringify({checks,errors},null,2));await browser.close(); }
  console.log(JSON.stringify({checks,errors},null,2));
})().catch(e=>{console.error(e);process.exitCode=1});
