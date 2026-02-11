/**
 * Vaporwave Grid Background Animation
 * Creates animated perspective grid with retro synthwave styling
 */

(function() {
  'use strict';

  // Configuration
  const config = {
    gridSize: 60,              // Size of grid cells (px)
    gridColor: 'rgba(255, 0, 255, 0.2)',    // Primary grid color
    gridColorAlt: 'rgba(0, 255, 255, 0.2)', // Secondary grid color
    animationSpeed: 4,          // Speed of animation (seconds)
    perspective: 400,           // Perspective value
    rotation: 80,               // X-axis rotation angle
    opacity: 0.3,               // Grid opacity
    height: '60vh',             // Grid height
  };

  // Create animated grid
  function createGrid() {
    // Remove existing grid if present
    const existingGrid = document.getElementById('vaporwave-grid');
    if (existingGrid) {
      existingGrid.remove();
    }

    const grid = document.createElement('div');
    grid.id = 'vaporwave-grid';
    grid.style.cssText = `
      position: fixed;
      bottom: 0;
      left: -50%;
      width: 200%;
      height: ${config.height};
      background-image:
        linear-gradient(${config.gridColor} 1px, transparent 1px),
        linear-gradient(90deg, ${config.gridColorAlt} 1px, transparent 1px);
      background-size: ${config.gridSize}px ${config.gridSize}px;
      transform: perspective(${config.perspective}px) rotateX(${config.rotation}deg);
      transform-origin: bottom center;
      pointer-events: none;
      z-index: -2;
      opacity: ${config.opacity};
      animation: gridScroll ${config.animationSpeed}s linear infinite;
    `;

    // Add keyframes if not already present
    if (!document.getElementById('vaporwave-grid-styles')) {
      const style = document.createElement('style');
      style.id = 'vaporwave-grid-styles';
      style.textContent = `
        @keyframes gridScroll {
          0% { background-position: 0 0; }
          100% { background-position: 0 ${config.gridSize}px; }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(grid);
  }

  // Add floating sun effect at bottom
  function createSun() {
    // Remove existing sun if present
    const existingSun = document.getElementById('vaporwave-sun');
    if (existingSun) {
      existingSun.remove();
    }

    const sun = document.createElement('div');
    sun.id = 'vaporwave-sun';
    sun.style.cssText = `
      position: fixed;
      bottom: -20%;
      left: 50%;
      transform: translateX(-50%);
      width: 80vw;
      height: 80vw;
      max-width: 600px;
      max-height: 600px;
      background: linear-gradient(to bottom, #FF9900, #FF00FF, #00FFFF);
      border-radius: 50%;
      filter: blur(80px);
      opacity: 0.15;
      pointer-events: none;
      z-index: -1;
      animation: sunPulse 6s ease-in-out infinite;
    `;

    // Add keyframes if not already present
    if (!document.getElementById('vaporwave-sun-styles')) {
      const style = document.createElement('style');
      style.id = 'vaporwave-sun-styles';
      style.textContent = `
        @keyframes sunPulse {
          0%, 100% {
            transform: translateX(-50%) scale(1);
            opacity: 0.15;
          }
          50% {
            transform: translateX(-50%) scale(1.05);
            opacity: 0.2;
          }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(sun);
  }

  // Handle visibility change to pause animations
  function handleVisibilityChange() {
    const grid = document.getElementById('vaporwave-grid');
    if (!grid) return;

    if (document.hidden) {
      grid.style.animationPlayState = 'paused';
    } else {
      grid.style.animationPlayState = 'running';
    }
  }

  // Initialize effects
  function init() {
    // Wait for page to load
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    createGrid();
    createSun();

    // Listen for visibility changes
    document.addEventListener('visibilitychange', handleVisibilityChange);

    console.log('[Vaporwave] Grid background initialized');
  }

  // Auto-initialize
  init();

  // Export for manual control if needed
  window.VaporwaveGrid = {
    config,
    createGrid,
    createSun,
    init
  };
})();
