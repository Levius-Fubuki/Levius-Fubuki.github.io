/* Start narrow diagram viewports on their readable column; keep all controls scoped. */
(()=>{
  function init(){
    document.querySelectorAll('.qwen-diagram[id^="wan-figure-"]').forEach(figure=>{
      const viewport=figure.querySelector('.qwen-diagram-viewport'),svg=figure.querySelector('svg');
      let lastWidth=0;
      function align(){
        const width=viewport.clientWidth;
        if(width<700){
          const left=figure.id==='wan-figure-3'?0:Math.max(0,(svg.getBoundingClientRect().width-width)/2);
          viewport.scrollTo({left,behavior:'instant'});
        }
        lastWidth=width;
      }
      new ResizeObserver(()=>{if(viewport.clientWidth!==lastWidth)align();}).observe(viewport);
      figure.querySelectorAll('[data-qwen-action]').forEach(button=>button.addEventListener('click',()=>requestAnimationFrame(align)));
      requestAnimationFrame(align);
    });
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
