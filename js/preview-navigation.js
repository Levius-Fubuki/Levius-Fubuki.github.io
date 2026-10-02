// Injected only by scripts/preview-server.py. Production HTML never loads this.
(() => {
  const revision=document.currentScript?.dataset.previewRevision;
  if(!revision||!['localhost','127.0.0.1','::1'].includes(location.hostname))return;
  function revise(link){
    const raw=link.getAttribute('href');if(!raw||raw.startsWith('#')||link.hasAttribute('download'))return;
    const url=new URL(raw,location.href);
    if(url.origin!==location.origin||!(/\/$|\.html$/.test(url.pathname)))return;
    url.searchParams.set('__preview',revision);link.href=url.href;
  }
  document.querySelectorAll('a[href]').forEach(revise);
  document.addEventListener('click',event=>{const link=event.target.closest('a[href]');if(link)revise(link)},{capture:true});
  new MutationObserver(records=>records.forEach(record=>record.addedNodes.forEach(node=>{
    if(!(node instanceof Element))return;
    if(node.matches('a[href]'))revise(node);node.querySelectorAll('a[href]').forEach(revise);
  }))).observe(document.body,{childList:true,subtree:true});
})();
