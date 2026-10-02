(() => {
  'use strict';
  const dataNode = document.getElementById('collection-reader-data');
  if (!dataNode) return;
  const data = JSON.parse(dataNode.textContent);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let active = null;
  const motion = () => {
    try { return typeof Element.prototype.animate==='function' && !reduced.matches && (window.LeviusMotion?.getState().enabled ?? localStorage.getItem('levius-ambient-motion') !== 'off'); }
    catch { return !reduced.matches; }
  };
  const node = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const remember = (id, page) => history.replaceState({...history.state,bookReader:id ? {id,page} : null},'',location.href);
  const animate = (state, element, frames, options) => {
    const animation = element.animate(frames, {...options,fill:'both'});
    state.animations.push(animation);
    return animation.finished.catch(()=>{});
  };
  function cleanup(state, {focus=true, keepState=false}={}) {
    if (active !== state) return;
    active = null;
    state.animations.forEach(animation=>animation.cancel());
    state.dialog.remove();
    document.documentElement.classList.remove('collection-is-reading');
    state.source.classList.remove('is-reading');
    state.source.setAttribute('aria-expanded','false');
    if (!keepState) remember(null,0);
    if (focus && state.source.isConnected) state.source.focus({preventScroll:true});
  }
  async function close({instant=false, focus=true}={}) {
    const state=active;
    if (!state || state.phase==='closing') return;
    const canAnimate = !instant && motion() && state.phase==='ready';
    state.phase='closing';
    state.dialog.querySelector('.reader-shell').inert=true;
    if (canAnimate) {
      state.dialog.querySelector('.reader-paper-content').inert=true;
      await animate(state,state.leaf,[{transform:`rotateY(${state.angle}deg)`},{transform:'rotateY(0deg)'}],{duration:360,easing:'ease-in-out'});
      if(active!==state)return;
      await Promise.all([
        animate(state,state.volume,[{transform:'none'},{transform:state.origin}],{duration:400,easing:'cubic-bezier(.5,0,.7,.4)'}),
        animate(state,state.dialog,[{opacity:1},{opacity:0}],{duration:300,delay:100})
      ]);
    }
    cleanup(state,{focus});
  }
  function renderPage(state, index, {focus=false}={}) {
    state.page=Math.max(0,Math.min(state.books.length,index));
    const content=state.dialog.querySelector('.reader-paper-content');
    content.replaceChildren(); content.scrollTop=0;
    const heading=node('h2','reader-page-title');heading.tabIndex=-1;
    if(state.page===0) {
      content.append(node('p','reader-eyebrow','CONTENTS / 合集目录'));
      heading.textContent=state.collection.title;content.append(heading);
      content.append(node('p','reader-page-description',state.collection.description));
      const list=node('ol','reader-index');
      state.books.forEach((book,index)=>{
        const li=node('li'),button=node('button','reader-index-entry');button.type='button';
        button.append(node('span','reader-index-number',String(index+1).padStart(2,'0')),node('span','reader-index-title',book.title),node('span','reader-index-arrow','↗'));
        button.addEventListener('click',()=>turn(state,index+1));li.append(button);list.append(li);
      });content.append(list);
    } else {
      const book=state.books[state.page-1];
      content.append(node('p','reader-eyebrow',`VOL. ${book.volume} / ${book.date.replaceAll('-','.')}`));
      const preview=node('div','reader-preview');
      const coverLink=node('a','reader-cover-link');coverLink.href=book.path;coverLink.setAttribute('aria-label',`阅读全文：${book.title}`);
      const original=document.querySelector(`[data-book="${book.id}"] .book-cover`);
      if(original){const cover=original.cloneNode(true);cover.querySelector('img').loading='eager';coverLink.append(cover);}
      const titleBlock=node('div','reader-preview-title');titleBlock.append(node('p','reader-preview-english',book.english));heading.textContent=book.title;titleBlock.append(heading);
      preview.append(coverLink,titleBlock);content.append(preview);
      content.append(node('p','reader-synopsis',book.summary));
      const read=node('a','reader-read','阅读全文 ↗');read.href=book.path;content.append(read);
      const indexButton=node('button','reader-to-index','返回目录');indexButton.type='button';indexButton.addEventListener('click',()=>turn(state,0));content.append(indexButton);
    }
    state.dialog.querySelector('.reader-page-number').textContent=`${String(state.page+1).padStart(2,'0')} / ${String(state.books.length+1).padStart(2,'0')}`;
    state.dialog.querySelector('[data-reader-step="-1"]').disabled=state.page===0;
    state.dialog.querySelector('[data-reader-step="1"]').disabled=state.page===state.books.length;
    remember(state.collection.id,state.page);
    if(focus)heading.focus({preventScroll:true});
  }
  async function turn(state,index) {
    if(active!==state||state.turning||state.phase!=='ready'||index===state.page||index<0||index>state.books.length)return;
    state.turning=true;
    const content=state.dialog.querySelector('.reader-paper-content');
    if(motion()) {
      // Like the reference, previews crossfade inside a stationary page block.
      await animate(state,content,[{opacity:1,transform:'translateX(0)'},{opacity:0,transform:`translateX(${index>state.page?-8:8}px)`}],{duration:110});
      if(active!==state){state.turning=false;return;}
      renderPage(state,index,{focus:true});
      await animate(state,content,[{opacity:0,transform:'translateX(8px)'},{opacity:1,transform:'translateX(0)'}],{duration:180});
    } else renderPage(state,index,{focus:true});
    state.turning=false;
  }
  async function open(source,{restore=false,page=0,onAll}={}) {
    if(active)return;
    if(typeof HTMLDialogElement==='undefined'||!HTMLDialogElement.prototype.showModal){onAll?.(source);return;}
    const collection=data.collections.find(item=>item.id===source.dataset.openCollection);
    if(!collection)return;
    window.LeviusDecode?.finish();
    const books=data.books.filter(book=>book.collection===collection.id);
    const dialog=node('dialog','collection-reader');dialog.setAttribute('aria-label',`${collection.title}合集，书籍预览`);
    dialog.innerHTML='<div class="reader-shell"><header class="reader-toolbar"><span class="reader-collection-label"></span><button type="button" class="reader-close">合上归架 <span aria-hidden="true">×</span></button></header><div class="reader-stage"><div class="reader-volume"><div class="reader-backboard"></div><div class="reader-volume-spine"></div><section class="reader-paper"><div class="reader-paper-content"></div><nav class="reader-pagination" aria-label="预览翻页"><button type="button" data-reader-step="-1" aria-label="上一页">←</button><span class="reader-page-number" aria-live="polite"></span><button type="button" data-reader-step="1" aria-label="下一页">→</button></nav></section><div class="reader-leaf"><div class="reader-front"></div><button type="button" class="reader-endpaper" aria-label="合上合集并放回书架"><img alt="" class="reader-endpaper-art" /><span class="reader-endpaper-border"></span><span class="reader-endpaper-type"><span class="reader-exlibris">EX LIBRIS / LEVIUS</span><span class="reader-mark"></span><strong></strong><span class="reader-subtitle"></span><span class="reader-imprint">COLLECTED NOTES</span></span></button></div></div></div><footer class="reader-footer"><span>← → 翻阅 <span class="reader-footer-hint">· ESC 合上</span></span><button type="button" class="reader-all">查看全部分卷 ↗</button></footer></div>';
    dialog.dataset.collection=collection.id;
    const front=source.querySelector('.collection-front').cloneNode(true);
    dialog.querySelector('.reader-front').append(front);
    dialog.querySelector('.reader-collection-label').textContent=`${collection.mark} ${collection.title} / ${collection.subtitle}`;
    dialog.querySelector('.reader-endpaper-art').src=books[0].art;
    dialog.querySelector('.reader-mark').textContent=collection.mark;
    dialog.querySelector('.reader-endpaper-type strong').textContent=collection.title;
    dialog.querySelector('.reader-subtitle').textContent=collection.subtitle;
    const sourceRect=source.querySelector('.spine-face').getBoundingClientRect();
    const state={dialog,source,collection,books,onAll,page:0,phase:'opening',turning:false,animations:[]};
    active=state;document.body.append(dialog);document.documentElement.classList.add('collection-is-reading');
    source.classList.add('is-reading');source.setAttribute('aria-expanded','true');dialog.showModal();
    state.volume=dialog.querySelector('.reader-volume');state.leaf=dialog.querySelector('.reader-leaf');
    state.angle=innerWidth<=600?-108:-160;
    const target=state.volume.getBoundingClientRect();
    const scale=sourceRect.height/target.height;
    state.origin=`translate(${sourceRect.left-target.left}px,${sourceRect.top-target.top+(sourceRect.height-target.height)/2}px) rotateY(90deg) scale(${scale})`;
    renderPage(state,page);
    dialog.querySelector('.reader-close').addEventListener('click',()=>close());
    dialog.querySelector('.reader-endpaper').addEventListener('click',()=>close());
    dialog.querySelector('.reader-all').addEventListener('click',async()=>{await close({instant:true,focus:false});onAll?.(source);});
    dialog.querySelectorAll('[data-reader-step]').forEach(button=>button.addEventListener('click',()=>turn(state,state.page+Number(button.dataset.readerStep))));
    dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
    // Page navigation can remove its clicked button synchronously in reduced motion.
    // Match the backdrop itself, never ancestry of a now-detached click target.
    dialog.addEventListener('click',event=>{if(event.target===dialog||event.target===dialog.querySelector('.reader-shell'))close();});
    dialog.addEventListener('keydown',event=>{
      if(event.altKey||event.ctrlKey||event.metaKey)return;
      if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();turn(state,state.page+(event.key==='ArrowRight'?1:-1));}
    });
    if(motion()&&!restore) {
      await Promise.all([
        animate(state,dialog,[{opacity:0},{opacity:1}],{duration:180}),
        animate(state,state.volume,[{transform:state.origin},{transform:'translateZ(45px) rotateY(0deg) scale(1.02)'},{transform:'none'}],{duration:650,easing:'cubic-bezier(.2,.75,.25,1)'}),
        animate(state,state.leaf,[{transform:'rotateY(0deg)'},{transform:`rotateY(${state.angle}deg)`}],{duration:650,delay:430,easing:'cubic-bezier(.2,.75,.25,1)'})
      ]);
    } else state.leaf.style.transform=`rotateY(${state.angle}deg)`;
    if(active!==state)return;
    state.phase='ready';dialog.dataset.ready='true';
    dialog.querySelector('.reader-page-title').focus({preventScroll:true});
  }
  addEventListener('pagehide',()=>{if(active)cleanup(active,{focus:false,keepState:true});});
  addEventListener('pageshow',event=>{
    if(event.persisted&&history.state?.bookReader){const state=history.state.bookReader;const source=document.querySelector(`[data-open-collection="${state.id}"]`);if(source)open(source,{restore:true,page:state.page,onAll:window.LeviusShowVolumes});}
  });
  addEventListener('resize',()=>{
    if(!active||active.phase!=='ready')return;
    const state=active;
    state.angle=innerWidth<=600?-108:-160;
    state.leaf.getAnimations().forEach(animation=>animation.cancel());
    state.leaf.style.transform=`rotateY(${state.angle}deg)`;
    const sourceRect=state.source.querySelector('.spine-face').getBoundingClientRect(),target=state.volume.getBoundingClientRect();
    state.origin=`translate(${sourceRect.left-target.left}px,${sourceRect.top-target.top+(sourceRect.height-target.height)/2}px) rotateY(90deg) scale(${sourceRect.height/target.height})`;
  });
  window.LeviusCollectionReader={open,close};
})();
