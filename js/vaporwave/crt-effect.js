/**
 * Vaporwave CRT Scanline Effect
 * Adds animated CRT scanline overlay to the page
 */

(function() {
  'use strict';

  // Configuration
  const config = {
    scanlineSpeed: 8,        // Speed of scanline animation (lower is faster)
    scanlineOpacity: 0.15,   // Opacity of scanline
    enableFlicker: true,     // Enable screen flicker effect
    flickerSpeed: 150,       // Speed of flicker (ms)
  };

  // Create scanline element
  function createScanline() {
    const scanline = document.createElement('div');
    scanline.id = 'vaporwave-crt-scanline';
    scanline.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 5px;
      background: linear-gradient(
        to bottom,
        transparent,
        rgba(0, 255, 255, 0.4),
        rgba(255, 0, 255, 0.4),
        transparent
      );
      opacity: ${config.scanlineOpacity};
      pointer-events: none;
      z-index: 10000;
      animation: scanlineMove ${config.scanlineSpeed}s linear infinite;
      box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    `;

    // Add keyframes if not already present
    if (!document.getElementById('vaporwave-scanline-styles')) {
      const style = document.createElement('style');
      style.id = 'vaporwave-scanline-styles';
      style.textContent = `
        @keyframes scanlineMove {
          0% { top: -5px; }
          100% { top: 100%; }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(scanline);
  }

  // Create screen flicker effect
  function createFlicker() {
    if (!config.enableFlicker) return;

    setInterval(() => {
      const opacity = 0.55 + Math.random() * 0.1;
      document.body.style.setProperty('--vaporwave-overlay-opacity', opacity);
    }, config.flickerSpeed);
  }

  // Initialize effects when DOM is ready
  function init() {
    // Wait for page to load
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    // Create effects
    createScanline();
    createFlicker();

    console.log('[Vaporwave] CRT effect initialized');
  }

  // Auto-initialize
  init();

  // Export for manual control if needed
  window.VaporwaveCRT = {
    config,
    createScanline,
    createFlicker,
    init
  };
})();
