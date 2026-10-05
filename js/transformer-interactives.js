// Only the five trusted standalone lesson shells can resize their article frames.
(() => {
  const frames = [...document.querySelectorAll('iframe.transformer-interactive')];
  window.addEventListener('message', event => {
    if (event.origin !== location.origin || event.data?.type !== 'transformer-viz-height') return;
    const frame = frames.find(item => item.contentWindow === event.source && item.dataset.widget === event.data.widget);
    const height = event.data.height;
    if (!frame || !Number.isFinite(height) || height < 100 || height > 5000) return;
    frame.style.height = `${Math.ceil(height)}px`;
  });
})();
