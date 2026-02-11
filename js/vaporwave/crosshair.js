/**
 * Crosshair Cursor Effect - Pure JS Implementation
 * Creates a custom crosshair cursor that follows mouse movement
 * with smooth lerp animation and shake effect on link hover
 */

(function() {
  'use strict';

  // Configuration
  const config = {
    lerpFactor: 0.15,            // Smooth interpolation factor
    shakeOnHover: true,          // Enable shake on link hover
    showDot: true                // Show center dot
  };

  // State
  let mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
  let current = { x: mouse.x, y: mouse.y };
  let isHoveringLink = false;
  let animationId = null;

  // DOM elements
  let container, lineH, lineV, dot;

  // Lerp function for smooth interpolation
  function lerp(start, end, factor) {
    return start + (end - start) * factor;
  }

  // Create crosshair elements
  function createCrosshair() {
    container = document.createElement('div');
    container.id = 'crosshair-container';

    // Horizontal line
    lineH = document.createElement('div');
    lineH.className = 'crosshair-line crosshair-horizontal';
    container.appendChild(lineH);

    // Vertical line
    lineV = document.createElement('div');
    lineV.className = 'crosshair-line crosshair-vertical';
    container.appendChild(lineV);

    // Center dot
    if (config.showDot) {
      dot = document.createElement('div');
      dot.className = 'crosshair-dot';
      container.appendChild(dot);
    }

    document.body.appendChild(container);

    // Show crosshair immediately
    lineH.classList.add('visible');
    lineV.classList.add('visible');
    if (dot) dot.classList.add('visible');
  }

  // Animation loop
  function animate() {
    // Smooth lerp to target position
    current.x = lerp(current.x, mouse.x, config.lerpFactor);
    current.y = lerp(current.y, mouse.y, config.lerpFactor);

    // Update line positions
    lineH.style.top = current.y + 'px';
    lineV.style.left = current.x + 'px';
    if (dot) {
      dot.style.left = current.x + 'px';
      dot.style.top = current.y + 'px';
    }

    animationId = requestAnimationFrame(animate);
  }

  // Mouse move handler
  function onMouseMove(e) {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  }

  // Add shake effect
  function addShake() {
    if (lineH) lineH.classList.add('shake');
    if (lineV) lineV.classList.add('shake');
  }

  // Remove shake effect
  function removeShake() {
    if (lineH) lineH.classList.remove('shake');
    if (lineV) lineV.classList.remove('shake');
  }

  // Setup link event listeners
  function setupLinkListeners() {
    // Find all interactive elements
    const interactiveElements = document.querySelectorAll('a, button, input, textarea, select, [onclick], [role="button"]');

    // Add event listeners to each element
    interactiveElements.forEach(el => {
      el.addEventListener('mouseenter', addShake);
      el.addEventListener('mouseleave', removeShake);
    });
  }

  // Initialize
  function init(options) {
    // Create crosshair elements
    createCrosshair();

    // Setup event listeners
    window.addEventListener('mousemove', onMouseMove);

    // Setup link hover effects
    if (config.shakeOnHover) {
      setupLinkListeners();
    }

    // Start animation
    animate();
  }

  // Destroy
  function destroy() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    window.removeEventListener('mousemove', onMouseMove);

    const interactiveElements = document.querySelectorAll('a, button, input, textarea, select, [onclick], [role="button"]');
    interactiveElements.forEach(el => {
      el.removeEventListener('mouseenter', addShake);
      el.removeEventListener('mouseleave', removeShake);
    });

    if (container && container.parentNode) {
      container.parentNode.removeChild(container);
    }

    isHoveringLink = false;
  }

  // Auto-initialize
  function autoInit() {
    // Check for disable attribute
    if (document.documentElement.getAttribute('data-crosshair') === 'false') {
      return;
    }

    init();
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }

  // Export to global scope
  window.Crosshair = {
    init,
    destroy
  };
})();
