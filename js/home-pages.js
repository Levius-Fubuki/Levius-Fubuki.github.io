(() => {
  'use strict';
  const main=document.querySelector('.site-shell main');
  if(!main)return;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const ids=['welcome','profile','terminal','articles'],labels=['开场','个人介绍','站内终端','文章索引'];
  const panels=ids.map(id=>{
    const section=document.getElementById(id),panel=document.createElement('section'),body=document.createElement('div');
    panel.className='page-panel';panel.dataset.page=id;panel.setAttribute('aria-label',labels[ids.indexOf(id)]);
    body.className='panel-body';section.before(panel);body.append(section);panel.append(body);return panel;
  });
  panels[3].firstElementChild.append(document.querySelector('.site-footer'));
  main.classList.add('page-stack');document.documentElement.classList.add('paged-home');
  const position=document.createElement('nav');position.className='page-position';position.setAttribute('aria-label','版面导航');
  const counter=document.createElement('span'),dots=document.createElement('div');dots.className='page-dots';position.append(counter,dots);document.body.append(position);
  let index=0,moving=false,scrollFrame=0,lastWheel=0,latched=false,wheelTotal=0,touch=null;
  const animated=()=>!reduced.matches&&window.LeviusMotion?.getState().enabled!==false;

  // Decode text nodes in place; never rebuild links, controls, or their listeners.
  const active=new Map(),seen=new WeakSet();let decodeFrame=0,lastDecode=0;
  const glyphs='01_/:+<>[]';const cjk='零壹数码字元解析';
  function excluded(node){return !node.parentElement||node.parentElement.closest('script,style,textarea,input,option,.sr-only,[data-no-decode],.terminal-input-path,.terminal-count,#terminal-path,#terminal-index-state');}
  function finishDecode(){active.forEach((state,node)=>{if(node.isConnected)node.nodeValue=state.original;state.parent.removeAttribute('data-decoding')});active.clear();document.querySelectorAll('[data-decoding]').forEach(node=>node.removeAttribute('data-decoding'));cancelAnimationFrame(decodeFrame);decodeFrame=0;}
  function decodeTick(now){
    decodeFrame=0;
    if(!animated()||document.hidden){finishDecode();return;}
    if(now-lastDecode>42){
      lastDecode=now;
      active.forEach((state,node)=>{
        if(!node.isConnected){state.parent.removeAttribute('data-decoding');active.delete(node);return;}
        const progress=Math.max(0,(now-state.start)/760);
        if(progress>=1){node.nodeValue=state.original;state.parent.removeAttribute('data-decoding');active.delete(node);return;}
        const count=Math.floor(progress*state.chars.length),tick=Math.floor(now/55);
        node.nodeValue=state.chars.map((char,i)=>{
          if(i<count||/\s/.test(char))return char;
          const alphabet=/[\u3400-\u9fff]/.test(char)?cjk:glyphs;
          return alphabet[(tick+i*7)%alphabet.length];
        }).join('');
      });
    }
    if(active.size)decodeFrame=requestAnimationFrame(decodeTick);
  }
  function decode(root,{repeat=false}={}){
    if(!root||!animated())return;
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let node,order=0;
    while((node=walker.nextNode())){
      if(excluded(node)||!node.nodeValue.trim()||active.has(node)||(!repeat&&seen.has(node)))continue;
      seen.add(node);const parent=node.parentElement;
      active.set(node,{original:node.nodeValue,chars:[...node.nodeValue],parent,start:performance.now()+Math.min(order++*14,180)});
      parent.setAttribute('data-decoding','true');
    }
    if(active.size&&!decodeFrame)decodeFrame=requestAnimationFrame(decodeTick);
  }
  window.LeviusDecode=Object.freeze({play:decode,finish:finishDecode});
  function update(){
    const header=document.querySelector('.site-header');header.hidden=index!==1;header.inert=index!==1;
    if(index===1)decode(header,{repeat:true});
    counter.textContent=`${String(index+1).padStart(2,'0')} / 04 — ${labels[index]}`;
    [...dots.children].forEach((button,i)=>button.setAttribute('aria-current',String(i===index)));
    panels.forEach((panel,i)=>{panel.inert=i!==index;panel.dataset.active=String(i===index)});
    document.body.dataset.page=ids[index];
    // Terminal's boot reveal triggers its own decode when the panels appear.
    if(index!==2)decode(panels[index],{repeat:true});
    else if(document.querySelector('#terminal').dataset.scene!=='booting')decode(panels[index],{repeat:true});
  }
  function go(next,{history=true,focus=false}={}){
    next=Math.max(0,Math.min(panels.length-1,next));
    if(moving)return;
    if(next===index&&Math.abs(main.scrollTop-next*main.clientHeight)<2)return;
    const from=main.scrollTop,to=next*main.clientHeight;index=next;moving=true;
    const header=document.querySelector('.site-header');header.hidden=true;header.inert=true;
    panels.forEach(panel=>{panel.inert=true});
    if(history)window.history.replaceState({},'',`#${ids[index]}`);
    const duration=animated()?720:0,start=performance.now();
    main.style.scrollSnapType='none';
    function tick(now){
      const t=duration?Math.min(1,(now-start)/duration):1,ease=t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
      main.scrollTop=from+(to-from)*ease;
      if(t<1){scrollFrame=requestAnimationFrame(tick);return;}
      main.style.scrollSnapType='';moving=false;update();
      if(focus){const target=panels[index].querySelector('input,button,a');target?.focus({preventScroll:true})}
    }
    cancelAnimationFrame(scrollFrame);scrollFrame=requestAnimationFrame(tick);
  }
  ids.forEach((id,i)=>{const button=document.createElement('button');button.type='button';button.setAttribute('aria-label',`${i+1} / ${labels[i]}`);button.addEventListener('click',()=>go(i));dots.append(button)});
  function nestedScroll(target,delta){
    // Logs and program results own the wheel even at their boundaries.
    if(target.closest('#terminal-output,#terminal-program-content,dialog'))return true;
    for(let node=target;node&&node!==main;node=node.parentElement){
      if(node.scrollHeight>node.clientHeight+2&&/(auto|scroll)/.test(getComputedStyle(node).overflowY)){
        if(delta>0&&node.scrollTop+node.clientHeight<node.scrollHeight-2)return true;
        if(delta<0&&node.scrollTop>2)return true;
      }
    }
    return false;
  }
  main.addEventListener('wheel',event=>{
    if(event.ctrlKey||Math.abs(event.deltaX)>Math.abs(event.deltaY)||document.querySelector('dialog[open]'))return;
    const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?main.clientHeight:1),now=performance.now();
    if(!moving&&nestedScroll(event.target,delta)){latched=false;lastWheel=now;return;}
    event.preventDefault();
    if(now-lastWheel>230){latched=false;wheelTotal=0}lastWheel=now;
    if(moving||latched)return;wheelTotal+=delta;
    if(Math.abs(wheelTotal)<20)return;
    latched=true;go(index+Math.sign(wheelTotal));
  },{passive:false});
  // Same-origin card iframe must participate in page scrolling without blocking
  // its existing click, hold-to-drag and right-click gestures.
  const card=document.querySelector('.identity-card'),wiredFrames=new WeakSet();
  function wireCardWheel(){
    const doc=card?.contentDocument;if(!doc||wiredFrames.has(doc))return;wiredFrames.add(doc);
    doc.addEventListener('wheel',event=>{
      const forwarded=new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaX:event.deltaX,deltaY:event.deltaY,deltaMode:event.deltaMode,ctrlKey:event.ctrlKey});
      if(!card.dispatchEvent(forwarded))event.preventDefault();
    },{passive:false});
  }
  card?.addEventListener('load',wireCardWheel);wireCardWheel();
  main.addEventListener('touchstart',event=>{if(event.touches.length===1)touch={y:event.touches[0].clientY,x:event.touches[0].clientX,target:event.target,handled:false}},{passive:true});
  main.addEventListener('touchmove',event=>{
    if(!touch||event.touches.length!==1)return;
    const delta=touch.y-event.touches[0].clientY;
    if(Math.abs(delta)<12||Math.abs(touch.x-event.touches[0].clientX)>Math.abs(delta)||touch.target.closest('input,textarea,dialog'))return;
    if(!moving&&nestedScroll(touch.target,delta))return;
    event.preventDefault();if(!touch.handled&&Math.abs(delta)>45){touch.handled=true;go(index+Math.sign(delta))}
  },{passive:false});
  main.addEventListener('touchend',()=>{touch=null},{passive:true});
  document.addEventListener('keydown',event=>{
    if(event.defaultPrevented||event.ctrlKey||event.metaKey||event.altKey||document.querySelector('dialog[open]')||event.target.closest('input,textarea,select,[contenteditable],#terminal-output,#terminal-program-content'))return;
    const directions={ArrowDown:1,PageDown:1,ArrowUp:-1,PageUp:-1,' ':event.shiftKey?-1:1};
    if(event.key===' '&&event.target.closest('button,a'))return;
    if(Object.hasOwn(directions,event.key)){
      if(nestedScroll(event.target,directions[event.key]))return;
      event.preventDefault();if(!event.repeat)go(index+directions[event.key]);
    }else if(event.key==='Home'||event.key==='End'){event.preventDefault();go(event.key==='Home'?0:3)}
  });
  document.addEventListener('click',event=>{
    const link=event.target.closest('a');if(!link||event.defaultPrevented||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||event.button!==0)return;
    const url=new URL(link.href,location.href);if(url.origin!==location.origin||url.pathname!==location.pathname)return;
    const target=ids.indexOf(url.hash.slice(1));if(target<0)return;
    event.preventDefault();go(target);
  });
  let scrollTimer;
  main.addEventListener('scroll',()=>{clearTimeout(scrollTimer);if(moving)return;scrollTimer=setTimeout(()=>{
    const next=Math.round(main.scrollTop/main.clientHeight);if(next!==index){index=next;update();history.replaceState({},'',`#${ids[index]}`)}
  },120)},{passive:true});
  function fit(){
    const terminal=document.getElementById('terminal');terminal.style.zoom='';
    if(innerWidth>1000){const style=getComputedStyle(panels[2]),height=panels[2].clientHeight-parseFloat(style.paddingTop)-parseFloat(style.paddingBottom);terminal.style.zoom=Math.min(1,height/terminal.offsetHeight)}
    cancelAnimationFrame(scrollFrame);moving=false;main.style.scrollSnapType='none';main.scrollTop=index*main.clientHeight;main.style.scrollSnapType='';update();
  }
  addEventListener('resize',fit);addEventListener('hashchange',()=>{const next=ids.indexOf(location.hash.slice(1));if(next>=0)go(next,{history:false})});
  reduced.addEventListener('change',()=>{if(reduced.matches)finishDecode()});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)finishDecode()});
  const mutations=new MutationObserver(records=>{
    records.forEach(record=>record.addedNodes.forEach(node=>{
      const parent=node.nodeType===Node.TEXT_NODE?node.parentElement:node;
      if(!(parent instanceof Element)||parent.closest('.page-position'))return;
      const panel=parent.closest('.page-panel');
      if((panel===panels[index]&&!moving)||parent.closest('dialog[open]'))decode(parent);
    }));
  });
  new MutationObserver(()=>{const dialog=document.querySelector('#search-dialog');if(dialog.open)decode(dialog,{repeat:true})}).observe(document.querySelector('#search-dialog'),{attributes:true,attributeFilter:['open']});
  mutations.observe(main,{childList:true,subtree:true});mutations.observe(document.querySelector('#search-results'),{childList:true,subtree:true});
  window.LeviusPages=Object.freeze({goTo:id=>go(ids.indexOf(id)),getState:()=>({page:ids[index],moving})});
  addEventListener('DOMContentLoaded',()=>{
    index=Math.max(0,ids.indexOf(location.hash.slice(1)));fit();
    new MutationObserver(()=>{if(!animated())finishDecode()}).observe(document.querySelector('.motion-toggle'),{attributes:true,attributeFilter:['aria-pressed']});
  },{once:true});
})();
