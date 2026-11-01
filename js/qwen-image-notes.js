/* No model inference: these are deterministic shape calculations and diagram controls. */
(() => {
  'use strict';
  function init() {
    document.querySelectorAll('.qwen-diagram').forEach(figure => {
      if (figure.dataset.qwenReady) return;
      figure.dataset.qwenReady = 'true';
      const svg = figure.querySelector('svg');
      const viewport = figure.querySelector('.qwen-diagram-viewport');
      const label = figure.querySelector('[data-qwen-zoom-label]');
      let zoom = 1;
      function update() {
        svg.style.width = `${740 * zoom}px`;
        label.textContent = `${Math.round(zoom * 100)}%`;
        figure.querySelector('[data-qwen-action="zoom-in"]').disabled = zoom >= 2;
        figure.querySelector('[data-qwen-action="zoom-out"]').disabled = zoom <= .75;
      }
      figure.querySelectorAll('[data-qwen-action]').forEach(button => {
        button.addEventListener('click', () => {
          switch (button.dataset.qwenAction) {
            case 'zoom-in': zoom = Math.min(2, zoom + .25); break;
            case 'zoom-out': zoom = Math.max(.75, zoom - .25); break;
            case 'reset': zoom = 1; viewport.scrollTo(0, 0); break;
            case 'expand': {
              const expanded = figure.classList.toggle('is-expanded');
              button.setAttribute('aria-expanded', String(expanded));
              button.textContent = expanded ? '收起高度' : '完整高度';
              break;
            }
          }
          update();
        });
      });
      update();
    });

    document.querySelectorAll('[data-qwen-image-lab]').forEach(root => {
      if (root.dataset.qwenReady) return;
      root.dataset.qwenReady = 'true';
      const get = name => root.querySelector(`[data-qwen-${name}]`);
      function render() {
        const resolution = Number(get('resolution').value);
        const world = Number(get('parallel').value);
        const edit = get('task').value === 'edit';
        const cached = get('phase').value === 'cached';
        const grid = resolution / 16;
        const tokens = grid * grid;
        const prefix = 128 + (edit ? tokens : 0);
        const queries = cached ? tokens : prefix + tokens;
        const keys = prefix + tokens;
        const heads = 32 / world;
        const intermediate = 12288 / world;
        const marker = `qwen-shape-arrow-${world}-${resolution}-${edit}-${cached}`;
        const node = (x, y, lines, accent = false) => `<g><rect class="qi-node${accent ? ' qi-node-accent' : ''}" x="${x}" y="${y}" width="220" height="76" rx="8"/><text x="${x + 110}" y="${y + (lines.length === 3 ? 22 : 30)}" text-anchor="middle">${lines.map((line, i) => `<tspan x="${x + 110}" dy="${i ? 21 : 0}">${line}</tspan>`).join('')}</text></g>`;
        const edge = (x1, y1, x2, y2) => `<path class="qi-edge" marker-end="url(#${marker})" d="M${x1} ${y1} L${x2} ${y2}"/>`;
        let graph = node(260, 15, ['目标 latent', `[1, ${tokens}, 64]`]);
        graph += edge(370, 91, 370, 125);
        graph += node(260, 125, [cached ? '缓存后续步的隐藏状态' : '首步统一序列', `[1, ${queries}, 4096]`]);
        if (world === 2) {
          graph += edge(300, 201, 130, 245) + edge(440, 201, 610, 245);
          graph += node(20, 245, ['rank 0：16 个头', `[1, ${queries}, 16, 128]`], true);
          graph += node(500, 245, ['rank 1：16 个头', `[1, ${queries}, 16, 128]`], true);
          graph += edge(130, 321, 300, 365) + edge(610, 321, 440, 365);
        } else {
          graph += edge(370, 201, 370, 245);
          graph += node(260, 245, ['单芯：32 个头', `[1, ${queries}, 32, 128]`], true);
          graph += edge(370, 321, 370, 365);
        }
        graph += node(260, 365, [world === 2 ? '输出投影 + AllReduce' : 'Attention 输出投影', `[1, ${queries}, 4096]`]);
        graph += edge(370, 441, 370, 485);
        graph += node(260, 485, [`每芯 SwiGLU：${intermediate} 通道`, `[1, ${queries}, ${intermediate}]`], true);
        graph += edge(370, 561, 370, 605);
        graph += node(260, 605, [world === 2 ? 'down + AllReduce + 残差' : 'down + 残差', `[1, ${queries}, 4096]`]);
        const stages = [
          ['目标 latent', `[1, ${tokens}, 64]`],
          [cached ? '缓存后续步隐藏状态' : '首步统一序列', `[1, ${queries}, 4096]`],
          [world === 2 ? 'rank 0 / rank 1 各自的 Q' : '单芯 Q', `[1, ${queries}, ${heads}, 128]`],
          [world === 2 ? '输出投影 + AllReduce' : 'Attention 输出投影', `[1, ${queries}, 4096]`],
          [`每芯 SwiGLU：${intermediate} 通道`, `[1, ${queries}, ${intermediate}]`],
          [world === 2 ? 'down + AllReduce + 残差' : 'down + 残差', `[1, ${queries}, 4096]`],
        ];
        const mobile = `<div class="qwen-shape-mobile">${stages.map(([label, shape]) => `<div class="qwen-shape-stage"><strong>${label}</strong><span>${shape}</span></div>`).join('<div class="qwen-shape-arrow" aria-hidden="true">↓</div>')}</div>`;
        get('shape-diagram').innerHTML = `<svg class="qwen-shape-desktop" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 700" role="img" aria-label="${resolution} 像素图片的 DiT 形状，${world} 芯执行"><title>DiT 形状与并行分工</title><defs><marker id="${marker}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs>${graph}</svg>${mobile}`;
        const detail = `${resolution}×${resolution}；latent 网格 ${grid}×${grid}；目标 N=${tokens}，条件前缀 P=${prefix}。当前 Q 长度=${queries}，K/V 长度=${keys}。每芯 ${heads} 个头，MLP 中间宽度 ${intermediate}；每层 ${world === 2 ? 2 : 0} 次 AllReduce。以上为形状推导，不是该配置的 GPU 实测。`;
        get('shape-detail').textContent = detail;
        root.dataset.targetTokens = String(tokens);
        root.dataset.queryTokens = String(queries);
        root.dataset.keyTokens = String(keys);
        root.dataset.localHeads = String(heads);
        root.dataset.localMlp = String(intermediate);
      }
      root.querySelectorAll('select').forEach(select => select.addEventListener('change', render));
      render();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
