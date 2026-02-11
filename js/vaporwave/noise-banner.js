/**
 * Noise Effect for Homepage Banner - Pure JS Implementation
 * Creates film grain/TV static noise effect on the homepage banner
 */

(function() {
  'use strict';

  // Configuration
  const config = {
    patternRefreshInterval: 2,    // Noise refresh interval (frames)
    patternAlpha: 10,             // Noise opacity (0-255)
    mixBlendMode: 'overlay',      // CSS blend mode
    canvasSize: 512               // Canvas size (power of 2)
  };

  // State
  let canvas = null;
  let ctx = null;
  let animationId = null;
  let frame = 0;
  let noiseData = null;
  let noise32 = null;
  let isInitialized = false;

  // Check if current page is homepage
  function isHomepage() {
    const path = window.location.pathname;
    return path === '/' || path === '/index.html' || path.endsWith('/');
  }

  // Create canvas element
  function createCanvas() {
    canvas = document.createElement('canvas');
    canvas.className = 'noise-banner-canvas';
    canvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      image-rendering: pixelated;
      mix-blend-mode: ${config.mixBlendMode};
      opacity: ${config.patternAlpha / 255};
      z-index: 2;
    `;

    // Find banner container
    const banner = document.querySelector('#page-header.full_page') ||
                   document.querySelector('.full_page') ||
                   document.querySelector('#page');

    if (!banner) {
      console.warn('Noise banner: banner container not found');
      return false;
    }

    banner.style.position = 'relative';
    banner.appendChild(canvas);

    return true;
  }

  // Resize canvas
  function resize() {
    if (!canvas) return;

    canvas.width = config.canvasSize;
    canvas.height = config.canvasSize;
  }

  // Initialize image data
  function initImageData() {
    if (!ctx) return;

    noiseData = ctx.createImageData(config.canvasSize, config.canvasSize);
    noise32 = new Uint32Array(noiseData.data.buffer);
  }

  // Draw noise grain
  function drawGrain() {
    if (!noise32) return;

    const a = config.patternAlpha << 24;
    for (let i = 0; i < noise32.length; i++) {
      const v = (Math.random() * 255) | 0;
      noise32[i] = a | (v << 16) | (v << 8) | v;
    }
  }

  // Animation loop
  function loop() {
    if (!ctx || !noiseData) return;

    if (frame % Math.max(1, Math.round(config.patternRefreshInterval)) === 0) {
      drawGrain();
      ctx.putImageData(noiseData, 0, 0);
    }

    frame++;
    animationId = requestAnimationFrame(loop);
  }

  // Initialize
  function init(options) {
    if (isInitialized) return;

    // Merge options
    if (options) {
      Object.keys(options).forEach(key => {
        if (config.hasOwnProperty(key)) {
          config[key] = options[key];
        }
      });
    }

    // Only apply to homepage
    if (!isHomepage()) {
      console.log('Noise banner: not on homepage, skipping');
      return;
    }

    // Create canvas
    if (!createCanvas()) {
      return;
    }

    // Get context
    ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) {
      console.error('Noise banner: failed to get canvas context');
      return;
    }

    // Setup
    resize();
    initImageData();
    drawGrain();
    ctx.putImageData(noiseData, 0, 0);

    // Start animation
    loop();

    // Handle resize
    window.addEventListener('resize', resize);

    isInitialized = true;
    console.log('Noise banner: initialized');
  }

  // Destroy
  function destroy() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    window.removeEventListener('resize', resize);

    if (canvas && canvas.parentNode) {
      canvas.parentNode.removeChild(canvas);
    }

    canvas = null;
    ctx = null;
    noiseData = null;
    noise32 = null;
    frame = 0;
    isInitialized = false;
  }

  // Re-initialize on page navigation (for SPA-like behavior)
  function handleNavigation() {
    destroy();
    if (isHomepage()) {
      setTimeout(init, 100);
    }
  }

  // Auto-initialize
  function autoInit() {
    // Check for disable attribute
    if (document.documentElement.getAttribute('data-noise-banner') === 'false') {
      return;
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      setTimeout(init, 100);
    }

    // Watch for page changes
    let lastUrl = location.href;
    new MutationObserver(() => {
      const url = location.href;
      if (url !== lastUrl) {
        lastUrl = url;
        handleNavigation();
      }
    }).observe(document, { subtree: true, childList: true });
  }

  // Start
  autoInit();

  // Export to global scope
  window.NoiseBanner = {
    init,
    destroy,
    config
  };
})();
