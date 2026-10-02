// Inline styles and bootstrap allow recovery before any external script loads.
(async () => {
  const modulePath = document.querySelector('[data-card-module]').dataset.cardModule;
  let objectURL;
  try {
    const blob = await fetchCardResource(modulePath, {
      onSource: source => { window.__cardRuntimeSource = source; }
    });
    objectURL = URL.createObjectURL(new Blob([blob], {type: 'text/javascript'}));
    await import(objectURL);
  } catch (error) {
    console.error('[card] Runtime unavailable:', error);
    document.querySelector('#loading')?.classList.add('error');
    const status = document.querySelector('[data-card-status]');
    if (status) status.textContent = 'Card resources are unavailable. Select Reload to try again.';
  } finally {
    if (objectURL) URL.revokeObjectURL(objectURL);
  }
})();
