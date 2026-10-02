(() => {
  'use strict';
  const terminal = document.querySelector('#terminal');
  if (!terminal) return;
  const stage = terminal.querySelector('.terminal-stage');
  const screen = terminal.querySelector('.terminal-screen');
  const program = terminal.querySelector('.terminal-program');
  const content = terminal.querySelector('#terminal-program-content');
  const transition = terminal.querySelector('#program-transition');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const animations = new Set();
  let phase = 'terminal', inView = false, booted = false, frame = 0, time = 0, lastFrame = 0;
  const wave = terminal.querySelector('#signal-wave'), ctx = wave.getContext('2d');
  const labels = {articles:'ARTICLES',reinforcement:'REINFORCE','deep-learning':'DEEP LEARNING',archives:'ARCHIVE',about:'LEVIUS'};
  const allowed = () => !reduced.matches && window.LeviusMotion?.getState().enabled !== false;
  const setPhase = value => { phase = value; terminal.dataset.scene = value; };
  function animate(node, frames, duration, delay = 0, easing = 'cubic-bezier(.445,.05,.55,.95)') {
    if (!allowed()) return Promise.resolve();
    const animation = node.animate(frames,{duration,delay,easing,fill:'backwards'});
    animations.add(animation);
    return animation.finished.catch(()=>{}).finally(()=>animations.delete(animation));
  }
  function finishAnimations() { animations.forEach(a=>a.finish()); }
  function revealFrame(node, inside = false) {
    if (!allowed() || innerWidth <= 700) return [];
    const rect = node.getBoundingClientRect(), bounds = stage.getBoundingClientRect();
    const make = (x,y,w,h) => {
      const mask = document.createElement('div');mask.className='frame-reveal-mask';
      Object.assign(mask.style,{position:'absolute',left:`${x-bounds.x}px`,top:`${y-bounds.y}px`,width:`${w}px`,height:`${h}px`,background:'#000',zIndex:inside?25:15,pointerEvents:'none'});
      stage.append(mask);return mask;
    };
    const a=make(rect.x-5,rect.y-5,rect.width+10,rect.height),b=make(rect.x-3,rect.y+5,rect.width,rect.height+10);
    const first=inside
      ? animate(a,[{height:`${rect.height}px`,width:`${rect.width+10}px`,transform:'translateX(0)'},{height:'10px',width:`${rect.width+10}px`,transform:'translateX(0)',offset:.5},{height:'10px',width:'0px',transform:`translateX(${rect.width}px)`}],1000,300)
      : animate(a,[{height:`${rect.height}px`,width:`${rect.width+10}px`},{height:'10px',width:`${rect.width+10}px`,offset:.5},{height:'10px',width:'0px'}],1000);
    const second=inside ? animate(b,[{width:`${rect.width}px`},{width:'0px'}],300)
      : animate(b,[{width:`${rect.width}px`,height:`${rect.height+10}px`},{width:'20px',height:`${rect.height+10}px`,offset:1/3},{width:'20px',height:`${rect.height+10}px`,offset:2/3},{width:'20px',height:'0px',transform:`translateY(${rect.height/2}px)`}],1500);
    return [first.finally(()=>a.remove()),second.finally(()=>b.remove())];
  }
  async function showTerminal({focus=false}={}) {
    if(phase==='transition'||phase==='booting')return;
    setPhase('booting');
    if(!program.hidden){program.inert=true;await animate(program,[{opacity:1},{opacity:0}],500)}
    program.hidden=true;transition.hidden=true;screen.hidden=false;screen.inert=true;
    const jobs=[...revealFrame(terminal.querySelector('.table-outside')),...revealFrame(terminal.querySelector('.table-inside'),true)];
    terminal.querySelectorAll('.lt-header').forEach(node=>jobs.push(animate(node,[{opacity:0},{opacity:1}],1,1500)));
    terminal.querySelectorAll('.lt-left,.lt-right').forEach(column=>{
      if(!allowed()||innerWidth<=700)return;
      for(const inner of [false,true]){
        const line=document.createElement('div');
        Object.assign(line.style,{position:'absolute',left:'0',right:'0',top:'25px',height:'525px',borderTop:'1px solid #303030',borderBottom:'1px solid #303030',pointerEvents:'none'});
        column.append(line);
        jobs.push(animate(line,[{top:'300px',height:'2px',opacity:0},{top:inner?'39px':'25px',height:'525px',opacity:1}],500,inner?1100:1000).finally(()=>line.remove()));
      }
    });
    terminal.querySelectorAll('.lt-panel').forEach((node,i)=>{
      const r=node.getBoundingClientRect();
      jobs.push(animate(node,[{width:'0px',height:'0px',transform:`translate(${r.width/2}px,${r.height/2}px)`,opacity:0},{width:`${r.width}px`,height:`${r.height}px`,transform:'translate(0,0)',opacity:1}],500,1500+i*100));
      [...node.querySelector('.lt-content').children].forEach(child=>jobs.push(animate(child,[{opacity:0},{opacity:1}],1,2200).then(()=>window.LeviusDecode?.play(child,{repeat:true}))));
    });
    jobs.push(animate(terminal.querySelector('#lt-keyboard'),[{opacity:0},{opacity:1}],2000,2000));
    const space=terminal.querySelector('[data-terminal-key=" "]');
    if(space)jobs.push(animate(space,[{transform:'scaleX(0)'},{transform:'scaleX(1)'}],500,2200));
    await Promise.all(jobs);screen.inert=false;setPhase('terminal');
    if(focus)terminal.querySelector('#terminal-input').focus({preventScroll:true});
  }
  async function fillProgram(type) {
    content.replaceChildren();
    const text = (tag,value) => {const node=document.createElement(tag);node.textContent=value;return node};
    if(type==='about'){
      content.append(text('h4',"Levius / L_F's Blog"),text('p','把想法写进代码。关注 AI Infra、多模态推理与 AI Agent，分享工程实践和学习笔记。'));
      const link=text('a','GitHub / Levius-Fubuki ↗');link.href='https://github.com/Levius-Fubuki';link.target='_blank';link.rel='noopener noreferrer';content.append(link);return;
    }
    content.append(text('p','正在读取文章索引…'));
    try {
      const articles=await window.LeviusBlog.loadIndex();
      const list=type==='reinforcement'?articles.filter(a=>/强化学习|基于价值|基于策略|Actor-Critic|蒙特卡洛/.test(a.title)):type==='deep-learning'?articles.filter(a=>/深度学习/.test(a.title)):articles;
      content.replaceChildren(text('p',`${list.length} 篇文章 / ${labels[type]}`));
      list.forEach(a=>{const row=text('article','');row.className='program-article';const link=text('a',`${String(articles.indexOf(a)+1).padStart(2,'0')} / ${a.title}`);link.href=a.url;row.append(link,text('p',a.text.slice(0,110)+'…'));content.append(row)});
    } catch {
      content.replaceChildren(text('p','文章索引加载失败。'));
      const retry=text('button','重试');retry.type='button';retry.addEventListener('click',()=>fillProgram(type));content.append(retry);
    }
  }
  async function launch(type='articles') {
    if(phase==='transition'||phase==='booting')return;
    if(!Object.hasOwn(labels,type))type='articles';
    setPhase('transition');screen.inert=true;
    transition.hidden=false;transition.firstElementChild.textContent=labels[type];
    terminal.querySelector('#terminal-program-title').textContent=`${labels[type]} / STREAM`;
    const data=fillProgram(type);
    await animate(screen,[{opacity:1},{opacity:0}],500);screen.hidden=true;
    await animate(transition.firstElementChild,[{opacity:1},{opacity:1}],2500);
    program.hidden=false;program.inert=true;window.LeviusDecode?.play(program,{repeat:true});
    await Promise.all([
      animate(transition.firstElementChild,[{opacity:1,transform:'scale(1)'},{opacity:0,transform:'scale(0)'}],600,0,'cubic-bezier(.68,-.55,.265,1.55)').then(()=>{transition.hidden=true}),
      animate(program,[{clipPath:'inset(0 100% 0 0)'},{clipPath:'inset(0 0 0 0)'}],1500,500), data
    ]);
    program.inert=false;setPhase('program');terminal.querySelector('.return-terminal').focus({preventScroll:true});
  }
  async function navigate(url,label='ARTICLE') {
    if(phase==='transition'||phase==='booting')return;
    setPhase('transition');const active=screen.hidden?program:screen;active.inert=true;
    transition.hidden=false;transition.firstElementChild.textContent=String(label).toUpperCase();
    await animate(active,[{opacity:1},{opacity:0}],500);active.hidden=true;
    await animate(transition.firstElementChild,[{opacity:1},{opacity:1}],2500);
    location.assign(url);
  }
  // Navigation and transitions belong only to this workbench.
  terminal.addEventListener('click',event=>{
    const link=event.target.closest('a');
    if(!link||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||link.target==='_blank')return;
    const url=new URL(link.href,location.href);
    if(url.origin!==location.origin||url.pathname===location.pathname)return;
    event.preventDefault();navigate(url.href);
  });
  terminal.querySelector('.return-terminal').addEventListener('click',()=>showTerminal({focus:true}));
  window.LeviusScene=Object.freeze({launch,navigate,returnToTerminal:()=>showTerminal({focus:true}),getState:()=>phase});
  function drawWave(){
    if(!ctx)return;ctx.clearRect(0,0,367,100);ctx.strokeStyle='#303030';ctx.lineWidth=.5;
    for(let i=0;i<25;i++){ctx.beginPath();ctx.moveTo(0,i*4);ctx.lineTo(367,i*4);ctx.stroke()}
    ctx.strokeStyle='#aaa';ctx.lineWidth=.65;ctx.beginPath();
    for(let i=0;i<500;i++){
      const x=i-250,a=(x+200)*(x+100)*(x+280)*(x+10)*(x-300)*(x-250)*(x-150)/1e14/1.5,p=2*Math.PI*(i%6)/6+i/500;
      const y=a*Math.sin(p)+5*Math.cos(i),z=a*Math.cos(p),ry=y*Math.cos(time)-z*Math.sin(time),rz=y*Math.sin(time)+z*Math.cos(time),perspective=110/(110-rz*.12),px=183.5+x*.735*perspective,py=50+ry*.65*perspective;
      if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
    }ctx.stroke();
  }
  function tick(now){frame=0;if(!inView||document.hidden||!allowed())return;frame=requestAnimationFrame(tick);if(now-lastFrame<33)return;time+=Math.min(66,now-lastFrame)*.0003;lastFrame=now;if(!screen.hidden)drawWave()}
  function sync(){cancelAnimationFrame(frame);const running=inView&&!document.hidden&&allowed();terminal.classList.toggle('terminal-motion-running',running);if(!allowed())finishAnimations();if(running)frame=requestAnimationFrame(tick)}
  reduced.addEventListener('change',sync);document.addEventListener('visibilitychange',sync);
  new MutationObserver(sync).observe(document.querySelector('.motion-toggle'),{attributes:true,attributeFilter:['aria-pressed']});
  new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;sync();if(inView&&!booted){booted=true;showTerminal()}},{threshold:.1}).observe(terminal);
  addEventListener('pageshow',event=>{if(!event.persisted)return;finishAnimations();transition.hidden=true;const isProgram=content.children.length>0;program.hidden=!isProgram;program.inert=!isProgram;screen.hidden=isProgram;screen.inert=isProgram;setPhase(isProgram?'program':'terminal')});
  setPhase('terminal');drawWave();
})();
