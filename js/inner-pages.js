(() => {
  'use strict';
  const status=document.getElementById('inner-status'),reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let motion=true;try{motion=localStorage.getItem('levius-ambient-motion')!=='off'}catch{}
  const active=new Map();let frame=0,lastTick=0;
  function finish(){active.forEach((item,node)=>{if(node.isConnected)node.nodeValue=item.original});active.clear();cancelAnimationFrame(frame);frame=0;}
  function tick(now){
    frame=0;if(reduced.matches||document.hidden||!motion){finish();return}
    if(now-lastTick>45){lastTick=now;active.forEach((item,node)=>{
      if(!node.isConnected){active.delete(node);return}
      const progress=Math.min(1,Math.max(0,(now-item.start)/660)),settled=Math.floor(progress*item.chars.length);
      if(progress===1){node.nodeValue=item.original;active.delete(node);return}
      node.nodeValue=item.chars.map((c,i)=>{if(i<settled||/\s/.test(c))return c;const glyphs='01_/:[]';return glyphs[(Math.floor(now/55)+i*3)%glyphs.length]}).join('');
    })}
    if(active.size)frame=requestAnimationFrame(tick);
  }
  function decode(element){
    if(reduced.matches||!motion||element.querySelector('.katex'))return;
    const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);let node;
    while((node=walker.nextNode())){
      if(!node.nodeValue.trim()||node.parentElement.closest('.sr-only,code,pre,.katex')||active.has(node))continue;
      active.set(node,{original:node.nodeValue,chars:[...node.nodeValue],start:performance.now()});
    }
    if(active.size&&!frame)frame=requestAnimationFrame(tick);
  }
  const reveal=new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(!entry.isIntersecting)return;reveal.unobserve(entry.target);decode(entry.target);
  }),{threshold:.15});
  document.querySelectorAll('.inner-title,.inner-description,.site-header .brand,.site-header .main-nav a,.article-sort-item-title,.article-sort-item.year,.category-list-link,.tag-cloud-list a,#article-container h2,#article-container h3,.gallery-empty h2').forEach(node=>reveal.observe(node));
  reduced.addEventListener('change',()=>{if(reduced.matches)finish()});document.addEventListener('visibilitychange',()=>{if(document.hidden)finish()});

  document.querySelectorAll('.library-nav a').forEach(link=>{if(link.pathname===location.pathname)link.setAttribute('aria-current','page')});
  const contents=document.querySelector('.contents-panel'),compact=matchMedia('(max-width:800px)');
  if(contents){contents.open=!compact.matches;compact.addEventListener('change',()=>{contents.open=!compact.matches})}
  const tocLinks=[...document.querySelectorAll('.toc-link')],headings=tocLinks.map(link=>{try{return document.getElementById(decodeURIComponent(link.hash.slice(1)))}catch{return null}});
  let scrollFrame=0;
  const progress=document.querySelector('.reading-progress'),topButton=document.querySelector('.back-top');
  function onScroll(){
    scrollFrame=0;const max=document.documentElement.scrollHeight-innerHeight;
    progress.style.transform=`scaleX(${max>0?Math.min(1,scrollY/max):0})`;topButton.hidden=scrollY<400;
    let current=0;headings.forEach((h,i)=>{if(h&&h.getBoundingClientRect().top<160)current=i});
    tocLinks.forEach((a,i)=>{if(i===current)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current')});
  }
  addEventListener('scroll',()=>{if(!scrollFrame)scrollFrame=requestAnimationFrame(onScroll)},{passive:true});onScroll();
  topButton.addEventListener('click',()=>scrollTo({top:0,behavior:reduced.matches?'instant':'smooth'}));
  const reading=document.querySelector('.reading-width');
  reading?.addEventListener('click',()=>{const enabled=document.body.classList.toggle('reading-wide');reading.setAttribute('aria-pressed',String(enabled));reading.textContent=enabled?'Exit focus mode':'Focus reading'});

  async function copy(text,button){
    const label=button.textContent;
    try{await navigator.clipboard.writeText(text);button.textContent='Copied';status.textContent='Copied to clipboard.'}
    catch{button.textContent='Copy failed';status.textContent='Clipboard unavailable. Select the text and copy it manually.'}
    setTimeout(()=>{button.textContent=label},1800);
  }
  document.querySelector('.copy-page')?.addEventListener('click',event=>copy(document.querySelector('link[rel=canonical]')?.href||location.href,event.currentTarget));
  document.querySelectorAll('#article-container figure.highlight').forEach(figure=>{
    figure.dataset.language=[...figure.classList].find(c=>c!=='highlight')||'code';
    const button=document.createElement('button');button.type='button';button.className='code-copy';button.textContent='Copy';button.setAttribute('aria-label','Copy code');
    button.addEventListener('click',()=>copy(figure.querySelector('.code pre')?.innerText||figure.querySelector('pre')?.innerText||'',button));figure.prepend(button);
  });
  document.querySelectorAll('#article-container table').forEach(table=>{
    if(table.closest('figure.highlight,.katex,.table-scroll'))return;
    const wrap=document.createElement('div');wrap.className='table-scroll';wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label','Horizontally scrollable data table');table.before(wrap);wrap.append(table);
  });
  const images=[...document.querySelectorAll('#article-container img')];
  if(images.length){
    const dialog=document.createElement('dialog');dialog.className='image-dialog';dialog.setAttribute('aria-label','Article image preview');
    const close=document.createElement('button');close.type='button';close.textContent='Close ×';const preview=document.createElement('img');dialog.append(close,preview);document.body.append(dialog);
    close.addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close()});
    images.forEach(img=>{
      if(img.closest('a'))return;
      img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label',`Enlarge image: ${img.alt||'Article image'}`);
      const open=()=>{preview.src=img.currentSrc||img.src;preview.alt=img.alt||'Article image';dialog.showModal()};
      img.addEventListener('click',open);img.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open()}});
    });
  }
})();
