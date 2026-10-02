// Inline in the card document, so a stalled bundle cannot block recovery.
(async () => {
  const modulePath = document.querySelector('[data-card-module]').dataset.cardModule;
  const local = new URL(modulePath, location.href);
  const mirror = new URL(local.pathname.slice(1), 'https://raw.githubusercontent.com/Levius-Fubuki/Levius-Fubuki.github.io/main/');
  const status = document.querySelector('[data-card-status]');
  let lastError;
  for (const [url, timeout] of [[local, 4000], [mirror, 12000]]) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    let objectURL;
    try {
      const response = await fetch(url, {signal: controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const code = await response.text();
      clearTimeout(timer);
      objectURL = URL.createObjectURL(new Blob([code], {type: 'text/javascript'}));
      await import(objectURL);
      window.__cardRuntimeSource = url.origin;
      return;
    } catch (error) {
      lastError = error;
      if (status) status.textContent = '正在切换卡片资源线路…';
    } finally {
      clearTimeout(timer);
      if (objectURL) URL.revokeObjectURL(objectURL);
    }
  }
  console.error('[card] Runtime unavailable:', lastError);
  document.querySelector('#loading')?.classList.add('error');
  if (status) status.textContent = '互动卡片资源暂时无法连接，请点击重新加载';
})();
