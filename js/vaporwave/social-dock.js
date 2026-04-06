/**
 * macOS-style Dock Effect for Social Share Buttons
 * Fixed version - safe scaling with proper bounds
 * Auto-add Font Awesome icons to buttons
 */

(function() {
  'use strict';

  // Safe configuration
  const CONFIG = {
    maxScale: 1.5,           // Maximum scale (1.5x size)
    proximity: 100,          // Distance in pixels to detect mouse
    baseScale: 1.0           // Normal scale
  };

  // Platform icon mapping (Font Awesome)
  const PLATFORM_ICONS = {
    facebook: 'fab fa-facebook-f',
    twitter: 'fab fa-twitter',
    weibo: 'fab fa-weibo',
    qq: 'fab fa-qq'
  };

  let mouseX = Infinity;
  let mouseY = Infinity;
  let dockButtons = [];

  // Add icon to button if missing
  function ensureIcon(button) {
    if (!button) return;

    const classList = Array.from(button.classList);
    let platform = null;

    // Detect platform from class name
    for (const cls of classList) {
      if (cls.startsWith('icon-')) {
        platform = cls.replace('icon-', '');
        break;
      }
    }

    if (!platform || !PLATFORM_ICONS[platform]) return;

    // Check if button already has content
    if (button.querySelector('i') || button.textContent.trim()) return;

    // Create icon element
    const icon = document.createElement('i');
    icon.className = PLATFORM_ICONS[platform];

    // Clear any text content and add icon
    button.innerHTML = '';
    button.appendChild(icon);
  }

  // Calculate distance from mouse point to element center
  function getDistanceToElement(mouseX, mouseY, element) {
    const rect = element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    return Math.sqrt(Math.pow(mouseX - centerX, 2) + Math.pow(mouseY - centerY, 2));
  }

  // Calculate scale based on distance (clamped to safe range)
  function calculateScale(distance) {
    if (distance >= CONFIG.proximity) return CONFIG.baseScale;
    if (distance <= 0) return CONFIG.maxScale;

    // Linear interpolation: closer = higher scale
    const normalizedDist = distance / CONFIG.proximity;
    const scaleRange = CONFIG.maxScale - CONFIG.baseScale;
    return CONFIG.maxScale - (scaleRange * normalizedDist);
  }

  // Update all button scales
  function updateScales() {
    dockButtons.forEach((button) => {
      if (!button || !document.body.contains(button)) return;

      // Ensure icon is present
      ensureIcon(button);

      const distance = getDistanceToElement(mouseX, mouseY, button);
      const scale = calculateScale(distance);

      // Clamp to safe range
      const clampedScale = Math.max(CONFIG.baseScale, Math.min(CONFIG.maxScale, scale));

      button.style.setProperty('--dock-scale', clampedScale.toFixed(3));
    });
  }

  // Throttled mouse move handler
  let rafId = null;
  function onMouseMove(e) {
    mouseX = e.clientX;
    mouseY = e.clientY;

    if (!rafId) {
      rafId = requestAnimationFrame(() => {
        updateScales();
        rafId = null;
      });
    }
  }

  // Initialize
  function init() {
    // Find all social share buttons with multiple selectors
    const selectors = [
      '.post-share .social-share .social-share-icon',
      '#post-share .social-share .social-share-icon',
      '.tag_share .social-share .social-share-icon',
      '.social-share .social-share-icon',
      'a.social-share-icon',
      '.social-share-icon'
    ];

    const found = new Set();
    selectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(el => found.add(el));
    });

    dockButtons = Array.from(found);

    if (dockButtons.length === 0) {
      console.log('[Social Dock] No buttons found');
      return false;
    }

    console.log(`[Social Dock] Initialized with ${dockButtons.length} buttons`);
    console.log('[Social Dock] Config:', CONFIG);

    // Add global mouse listener
    document.addEventListener('mousemove', onMouseMove, { passive: true });

    // Initial update (this will also add icons)
    updateScales();

    return true;
  }

  // Wait for DOM to be ready and sharejs to generate buttons
  function waitForButtons() {
    if (init()) return;

    const observer = new MutationObserver(() => {
      if (init()) observer.disconnect();
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Timeout fallback
    setTimeout(() => {
      observer.disconnect();
      init();
    }, 3000);
  }

  // Start
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForButtons);
  } else {
    waitForButtons();
  }

  // Export for debugging
  window.SocialDock = {
    CONFIG,
    init,
    updateScales
  };
})();
