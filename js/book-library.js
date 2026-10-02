(() => {
  'use strict';
  document.documentElement.classList.add('book-ui-ready');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const motionEnabled = () => {
    try { return window.LeviusMotion?.getState().enabled ?? localStorage.getItem('levius-ambient-motion') !== 'off'; }
    catch { return true; }
  };
  const rail = document.querySelector('.book-rail');
  if (rail) {
    const books = [...rail.querySelectorAll('.shelf-book')];
    const controls = [...document.querySelectorAll('[data-shelf-step]')];
    const cabinet = document.querySelector('.collection-shelf');
    const detail = document.querySelector('.collection-detail');
    const spines = [...document.querySelectorAll('[data-open-collection]')];
    const back = document.querySelector('.collection-return');
    let selected = null, switching = false;
    const motion = () => !reduced.matches && motionEnabled();
    const update = () => controls.forEach(button => {
      button.disabled = Number(button.dataset.shelfStep) < 0
        ? rail.scrollLeft < 2
        : rail.scrollLeft + rail.clientWidth >= rail.scrollWidth - 2;
    });
    function move(direction) {
      const first = books.find(book => !book.hidden);
      if (!first) return;
      const stride = first.getBoundingClientRect().width + parseFloat(getComputedStyle(rail).gap);
      const amount = Math.max(1, Math.floor(rail.clientWidth / stride)) * stride;
      rail.scrollBy({left: direction * amount, behavior: motion() ? 'smooth' : 'instant'});
    }
    controls.forEach(button => button.addEventListener('click', () => move(Number(button.dataset.shelfStep))));
    async function switchView(from, to, restore) {
      from.inert = true;
      if (motion() && !restore) {
        await from.animate([{opacity:1,transform:'translateY(0)'},{opacity:0,transform:'translateY(10px)'}], {duration:150,easing:'ease-in',fill:'none'}).finished;
      }
      from.hidden = true; from.inert = false; to.hidden = false;
      if (motion() && !restore) {
        to.animate([{opacity:0,transform:'translateY(14px)'},{opacity:1,transform:'translateY(0)'}], {duration:320,easing:'cubic-bezier(.2,.75,.25,1)'});
      }
    }
    async function openCollection(button, {restore=false}={}) {
      if (switching || !button) return;
      switching = true; selected = button;
      window.LeviusDecode?.finish();
      const id = button.dataset.openCollection;
      books.forEach(book => { book.hidden = book.dataset.series !== id; });
      spines.forEach(spine => spine.setAttribute('aria-expanded', String(spine===button)));
      document.getElementById('collection-volume-title').textContent = button.dataset.title;
      document.getElementById('collection-volume-subtitle').textContent = button.dataset.subtitle;
      document.querySelector('.collection-description-line').textContent = button.dataset.description;
      document.querySelector('.collection-index-link').href = button.dataset.href;
      document.querySelector('.shelf-count').textContent = `${String(books.filter(book=>!book.hidden).length).padStart(2,'0')} VOLUMES`;
      history.replaceState({...history.state,bookCollection:id},'',location.href);
      await switchView(cabinet, detail, restore);
      rail.scrollTo({left:0,behavior:'instant'}); update();
      if (!restore) document.getElementById('collection-volume-title').focus({preventScroll:true});
      switching = false;
    }
    async function returnToCabinet() {
      if (switching || detail.hidden) return;
      switching = true;
      window.LeviusDecode?.finish();
      await switchView(detail, cabinet, false);
      spines.forEach(spine=>spine.setAttribute('aria-expanded','false'));
      history.replaceState({...history.state,bookCollection:null},'',location.href);
      selected?.focus({preventScroll:true}); switching = false;
    }
    window.LeviusShowVolumes = openCollection;
    spines.forEach(button=>button.addEventListener('click',()=>window.LeviusCollectionReader ? window.LeviusCollectionReader.open(button,{onAll:openCollection}) : openCollection(button)));
    back.addEventListener('click',returnToCabinet);
    detail.addEventListener('keydown',event=>{
      if(event.key==='Escape'&&!document.querySelector('dialog[open]')) {event.preventDefault();returnToCabinet();}
    });
    const remembered = spines.find(button=>button.dataset.openCollection===history.state?.bookCollection);
    const readerState=history.state?.bookReader;
    const readerSource=spines.find(button=>button.dataset.openCollection===readerState?.id);
    if(readerSource && window.LeviusCollectionReader) window.LeviusCollectionReader.open(readerSource,{restore:true,page:readerState.page,onAll:openCollection});
    else if (remembered) openCollection(remembered,{restore:true});
    rail.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      move(event.key === 'ArrowRight' ? 1 : -1);
    });
    rail.addEventListener('scroll', update, {passive: true});
    new ResizeObserver(update).observe(rail);
    update();
  }

  document.querySelectorAll('.has-book-preview').forEach(row => {
    const button = row.querySelector('.book-preview-toggle');
    const preview = row.querySelector('.archive-book-preview');
    let pinned = false;
    const setOpen = open => {
      row.classList.toggle('is-preview-open', open);
      button.setAttribute('aria-expanded', String(open));
      button.querySelector('.preview-sign').textContent = open ? '−' : '＋';
      preview.inert = !open;
    };
    row.addEventListener('pointerenter', event => { if (event.pointerType === 'mouse') setOpen(true); });
    row.addEventListener('pointerleave', () => { if (!pinned && !row.contains(document.activeElement)) setOpen(false); });
    row.addEventListener('focusin', () => setOpen(true));
    row.addEventListener('focusout', event => {
      if (!pinned && !row.contains(event.relatedTarget) && !row.matches(':hover')) setOpen(false);
    });
    button.addEventListener('click', () => { pinned = !pinned; setOpen(pinned); });
    row.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      pinned = false;
      if (preview.contains(document.activeElement)) button.focus();
      setOpen(false);
    });
  });

  // Keep ordinary links as the fallback; only same-tab cover clicks get a page turn.
  let opening = null;
  function clearOpening() {
    if (!opening) return;
    clearTimeout(opening.timeout);
    opening.animations.forEach(animation => animation.cancel());
    opening.dialog.remove();
    document.documentElement.classList.remove('book-is-opening');
    opening = null;
  }
  addEventListener('pageshow', clearOpening);
  addEventListener('pagehide', clearOpening);
  document.addEventListener('click', event => {
    const link = event.target.closest?.('a.book-link, a.archive-cover-link');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
    const destination = new URL(link.href, location.href);
    if (destination.origin !== location.origin || reduced.matches || !motionEnabled() || typeof HTMLDialogElement === 'undefined' || !Element.prototype.animate) return;
    if (opening) { event.preventDefault(); return; }
    event.preventDefault();
    window.LeviusDecode?.finish();
    const cover = link.querySelector('.book-cover');
    if (!cover) { location.assign(destination.href); return; }
    const title = (link.getAttribute('aria-label') || 'Article').replace(/^(?:Read|阅读)[:：]\s*/i, '');
    const dialog = document.createElement('dialog');
    dialog.className = 'book-opening';
    dialog.setAttribute('aria-label', `Opening ${title}`);
    dialog.innerHTML = '<div class="book-opening-stage" aria-hidden="true"><div class="book-opening-volume"><div class="book-opening-paper"><span class="book-opening-edition">LEVIUS / COLLECTED NOTES</span><span class="book-opening-ornament">✧</span><strong></strong><span class="book-opening-rules"></span><span class="book-opening-author">LEVIUS FUBUKI</span></div><div class="book-turning-leaf leaf-two"></div><div class="book-turning-leaf leaf-one"></div><div class="book-turning-cover"><div class="book-opening-inside"><span>EX LIBRIS<br>LEVIUS FUBUKI</span></div></div></div></div><p class="book-opening-status" role="status">Opening <span></span></p>';
    dialog.querySelector('.book-opening-paper strong').textContent = title;
    dialog.querySelector('.book-opening-status span').textContent = title;
    const duplicate = cover.cloneNode(true);
    duplicate.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
    duplicate.querySelectorAll('[data-decoding]').forEach(node => node.removeAttribute('data-decoding'));
    dialog.querySelector('.book-turning-cover').append(duplicate);
    document.body.append(dialog);
    opening = {dialog, animations:[], timeout:0};
    const current = opening;
    let navigating = false;
    const navigate = () => {
      if (navigating || opening !== current) return;
      navigating = true;
      clearTimeout(current.timeout);
      location.assign(destination.href);
    };
    dialog.addEventListener('cancel', event => { event.preventDefault(); navigate(); });
    try {
      document.documentElement.classList.add('book-is-opening');
      dialog.showModal();
      const animate = (selector, frames, duration, delay = 0) => {
        const element = selector ? dialog.querySelector(selector) : dialog;
        const animation = element.animate(frames, {duration, delay, easing:'cubic-bezier(.22,.72,.18,1)', fill:'both'});
        current.animations.push(animation);
        return animation;
      };
      animate(null, [{opacity:0}, {opacity:1}], 180);
      animate('.book-opening-volume', [
        {transform:'translateX(-50%) translateY(28px) rotateX(8deg) scale(.8)'},
        {transform:'translateX(0) translateY(0) rotateX(0deg) scale(1)'}
      ], 760);
      animate('.book-turning-cover', [{transform:'rotateY(0deg)'}, {transform:'rotateY(-160deg)'}], 760, 180);
      animate('.leaf-one', [{transform:'rotateY(0deg)'}, {transform:'rotateY(-147deg)'}], 660, 330);
      const last = animate('.leaf-two', [{transform:'rotateY(0deg)'}, {transform:'rotateY(-135deg)'}], 680, 480);
      animate('.book-opening-status', [{opacity:0,transform:'translateY(6px)'},{opacity:1,transform:'translateY(0)'}], 360, 300);
      last.finished.then(navigate).catch(() => {});
      // A cancelled/throttled animation must never leave an otherwise valid link stuck.
      current.timeout = setTimeout(navigate, 1650);
    } catch {
      clearOpening();
      location.assign(destination.href);
    }
  });
})();
