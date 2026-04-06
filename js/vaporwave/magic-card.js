/**
 * Magic Card Effects - Pure JS Implementation
 * Adds interactive effects to cards: particles, spotlight, tilt, magnetism, ripple
 * Applied to article cards and sidebar cards
 */

(function() {
  'use strict';

  // Configuration
  const config = {
    // Colors
    glowColor: '255, 0, 255',           // Theme color (Magenta - RGB)

    // Particle settings
    enableParticles: true,              // Enable particle effect on hover
    particleCount: 12,                  // Number of particles per card
    particleSize: 4,                    // Particle size (px)

    // Spotlight settings
    enableSpotlight: true,              // Enable global spotlight effect
    spotlightRadius: 300,               // Spotlight influence radius

    // Interaction settings
    enableTilt: true,                   // Enable 3D tilt effect
    enableMagnetism: true,              // Enable magnetic effect
    enableRipple: true,                 // Enable click ripple effect
    enableBorderGlow: true,             // Enable border glow effect

    // Animation settings
    tiltStrength: 10,                   // Maximum tilt angle (degrees)
    magnetismStrength: 0.05,            // Magnetism movement strength (0-1)
    animationSpeed: 0.1,                // Animation duration (seconds)

    // Performance
    throttleDelay: 16,                   // Mouse move throttle (ms) ~60fps

    // Selectors - split into array for proper matching
    // EXCLUDED: '.post-content', 'article' (article content should NOT have 3D effects)
    cardSelectors: [
      '.recent-post-item',
      '.card-widget',
      '.card-author',
      '.card-announcement',
      '.card-recent-post',
      '.card-categories',
      '.card-tags',
      '.card-archives',
      '.card-webinfo',
      '.card-post-series'
    ]
  };

  // State
  let spotlightElement = null;
  let cards = [];
  let animationFrameId = null;
  let lastMouseMoveTime = 0;
  let mouseX = 0;
  let mouseY = 0;

  // Utility: Throttle function
  function throttle(func, delay) {
    let lastCall = 0;
    return function(...args) {
      const now = Date.now();
      if (now - lastCall >= delay) {
        lastCall = now;
        return func.apply(this, args);
      }
    };
  }

  // Utility: Lerp for smooth animation
  function lerp(start, end, factor) {
    return start + (end - start) * factor;
  }

  // Check if element matches any of our card selectors
  function isCardElement(element) {
    if (!element || !element.matches) return false;
    return config.cardSelectors.some(selector => {
      try {
        return element.matches(selector);
      } catch (e) {
        return false;
      }
    });
  }

  // Create particle element
  function createParticle(x, y, color) {
    const particle = document.createElement('div');
    particle.className = 'magic-particle';
    particle.style.cssText = `
      position: absolute;
      width: ${config.particleSize}px;
      height: ${config.particleSize}px;
      border-radius: 50%;
      background: rgba(${color}, 1);
      box-shadow: 0 0 6px rgba(${color}, 0.6);
      pointer-events: none;
      z-index: 100;
      left: ${x}px;
      top: ${y}px;
      opacity: 0;
      transform: scale(0);
      transition: opacity 0.3s ease, transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    `;
    return particle;
  }

  // Animate particle
  function animateParticle(particle) {
    // Random movement
    const moveX = (Math.random() - 0.5) * 100;
    const moveY = (Math.random() - 0.5) * 100;
    const rotation = Math.random() * 360;

    // Fade in and scale up
    requestAnimationFrame(() => {
      particle.style.opacity = '1';
      particle.style.transform = 'scale(1)';

      // Start continuous animation
      const duration = 2000 + Math.random() * 2000;
      particle.animate([
        { transform: 'scale(1) translate(0, 0) rotate(0deg)', opacity: 1 },
        { transform: `scale(1) translate(${moveX}px, ${moveY}px) rotate(${rotation}deg)`, opacity: 0.3 }
      ], {
        duration: duration,
        iterations: Infinity,
        direction: 'alternate',
        easing: 'ease-in-out'
      });
    });
  }

  // Create ripple effect
  function createRipple(card, x, y) {
    const rect = card.getBoundingClientRect();
    const clickX = x - rect.left;
    const clickY = y - rect.top;

    const maxDistance = Math.max(
      Math.hypot(clickX, clickY),
      Math.hypot(clickX - rect.width, clickY),
      Math.hypot(clickX, clickY - rect.height),
      Math.hypot(clickX - rect.width, clickY - rect.height)
    );

    const ripple = document.createElement('div');
    ripple.className = 'magic-ripple';
    ripple.style.cssText = `
      position: absolute;
      width: ${maxDistance * 2}px;
      height: ${maxDistance * 2}px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(${config.glowColor}, 0.4) 0%, rgba(${config.glowColor}, 0.2) 30%, transparent 70%);
      left: ${clickX - maxDistance}px;
      top: ${clickY - maxDistance}px;
      pointer-events: none;
      z-index: 1000;
      transform: scale(0);
      opacity: 1;
    `;

    card.appendChild(ripple);

    // Animate ripple
    ripple.animate([
      { transform: 'scale(0)', opacity: 1 },
      { transform: 'scale(1)', opacity: 0 }
    ], {
      duration: 800,
      easing: 'ease-out'
    }).onfinish = () => ripple.remove();
  }

  // Update card glow properties
  function updateCardGlow(card, mouseX, mouseY, intensity) {
    const rect = card.getBoundingClientRect();
    const relativeX = ((mouseX - rect.left) / rect.width) * 100;
    const relativeY = ((mouseY - rect.top) / rect.height) * 100;

    card.style.setProperty('--magic-glow-x', `${relativeX}%`);
    card.style.setProperty('--magic-glow-y', `${relativeY}%`);
    // Calculate rgba values directly instead of using calc()
    const alpha1 = intensity * 0.8;
    const alpha2 = intensity * 0.4;
    card.style.setProperty('--magic-glow-color-1', `rgba(${config.glowColor}, ${alpha1})`);
    card.style.setProperty('--magic-glow-color-2', `rgba(${config.glowColor}, ${alpha2})`);
    card.style.setProperty('--magic-glow-radius', `${config.spotlightRadius}px`);
  }

  // Setup spotlight effect
  function setupSpotlight() {
    if (!config.enableSpotlight || spotlightElement) return;

    spotlightElement = document.createElement('div');
    spotlightElement.className = 'magic-spotlight';
    spotlightElement.style.cssText = `
      position: fixed;
      width: 800px;
      height: 800px;
      border-radius: 50%;
      pointer-events: none;
      background: radial-gradient(circle,
        rgba(${config.glowColor}, 0.15) 0%,
        rgba(${config.glowColor}, 0.08) 15%,
        rgba(${config.glowColor}, 0.04) 25%,
        rgba(${config.glowColor}, 0.02) 40%,
        rgba(${config.glowColor}, 0.01) 65%,
        transparent 70%
      );
      z-index: 200;
      opacity: 0;
      transform: translate(-50%, -50%);
      mix-blend-mode: screen;
      transition: opacity 0.3s ease, left 0.1s ease, top 0.1s ease;
    `;

    document.body.appendChild(spotlightElement);
  }

  // Handle global mouse move for spotlight
  const handleGlobalMouseMove = throttle((e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    lastMouseMoveTime = Date.now();

    if (spotlightElement && config.enableSpotlight) {
      spotlightElement.style.left = mouseX + 'px';
      spotlightElement.style.top = mouseY + 'px';

      // Update card glows based on proximity
      cards.forEach(card => {
        if (!card.element) return;

        const rect = card.element.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const distance = Math.hypot(mouseX - centerX, mouseY - centerY) - Math.max(rect.width, rect.height) / 2;
        const effectiveDistance = Math.max(0, distance);

        const proximity = config.spotlightRadius * 0.5;
        const fadeDistance = config.spotlightRadius * 0.75;

        let glowIntensity = 0;
        if (effectiveDistance <= proximity) {
          glowIntensity = 1;
        } else if (effectiveDistance <= fadeDistance) {
          glowIntensity = (fadeDistance - effectiveDistance) / (fadeDistance - proximity);
        }

        updateCardGlow(card.element, mouseX, mouseY, glowIntensity);
      });
    }
  }, config.throttleDelay);

  // Setup card effects
  function setupCardEffects(card) {
    if (!card || card._magicCardHandlers) return;

    // Ensure card has magic-card class
    card.classList.add('magic-card');
    card.classList.add('magic-card--glow');

    let particles = [];
    let particleTimeouts = [];
    let currentTiltX = 0;
    let currentTiltY = 0;
    let currentMagnetX = 0;
    let currentMagnetY = 0;

    const handleMouseEnter = () => {
      card.classList.add('magic-card--hover');

      // Generate particles
      if (config.enableParticles) {
        const rect = card.getBoundingClientRect();
        for (let i = 0; i < config.particleCount; i++) {
          const timeoutId = setTimeout(() => {
            if (!card.classList.contains('magic-card--hover') || !card.parentNode) return;

            const x = Math.random() * rect.width;
            const y = Math.random() * rect.height;
            const particle = createParticle(x, y, config.glowColor);

            card.appendChild(particle);
            particles.push(particle);
            animateParticle(particle);
          }, i * 100);

          particleTimeouts.push(timeoutId);
        }
      }

      // Initial lift
      card.style.transform = 'translateY(-2px)';
    };

    const handleMouseMove = (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      let transforms = [];

      // Tilt effect (mimicking Profile Card.vue style)
      if (config.enableTilt) {
        const targetTiltX = ((x - centerX) / centerX) * -config.tiltStrength;  // X position controls X-axis rotation
        const targetTiltY = ((y - centerY) / centerY) * config.tiltStrength;   // Y position controls Y-axis rotation

        currentTiltX = lerp(currentTiltX, targetTiltX, 0.1);
        currentTiltY = lerp(currentTiltY, targetTiltY, 0.1);

        transforms.push(`rotateX(${currentTiltX}deg) rotateY(${currentTiltY}deg)`);
      }

      // Magnetism effect
      if (config.enableMagnetism) {
        const targetMagnetX = (x - centerX) * config.magnetismStrength;
        const targetMagnetY = (y - centerY) * config.magnetismStrength;

        currentMagnetX = lerp(currentMagnetX, targetMagnetX, 0.1);
        currentMagnetY = lerp(currentMagnetY, targetMagnetY, 0.1);

        transforms.push(`translate(${currentMagnetX}px, ${currentMagnetY - 2}px)`);
      } else {
        transforms.push('translateY(-2px)');
      }

      card.style.transform = transforms.join(' ');

      // Update glow position
      if (config.enableBorderGlow) {
        updateCardGlow(card, e.clientX, e.clientY, 1);
      }
    };

    const handleMouseLeave = () => {
      card.classList.remove('magic-card--hover');

      // Clear particles
      particleTimeouts.forEach(clearTimeout);
      particleTimeouts = [];

      particles.forEach(particle => {
        if (particle.parentNode) {
          particle.style.opacity = '0';
          particle.style.transform = 'scale(0)';
          setTimeout(() => {
            if (particle.parentNode) particle.remove();
          }, 300);
        }
      });
      particles = [];

      // Reset transforms
      currentTiltX = 0;
      currentTiltY = 0;
      currentMagnetX = 0;
      currentMagnetY = 0;

      card.style.transform = '';
      card.style.transition = 'transform 0.3s ease';

      setTimeout(() => {
        card.style.transition = '';
      }, 300);
    };

    const handleClick = (e) => {
      if (!config.enableRipple) return;
      createRipple(card, e.clientX, e.clientY);
    };

    // Store event handlers for cleanup
    card._magicCardHandlers = {
      mouseenter: handleMouseEnter,
      mousemove: handleMouseMove,
      mouseleave: handleMouseLeave,
      click: handleClick
    };

    card.addEventListener('mouseenter', handleMouseEnter);
    card.addEventListener('mousemove', handleMouseMove);
    card.addEventListener('mouseleave', handleMouseLeave);
    card.addEventListener('click', handleClick);

    // Add perspective for 3D effect
    if (config.enableTilt) {
      card.style.perspective = '1000px';
      card.style.transformStyle = 'preserve-3d';
    }
  }

  // Remove card effects
  function removeCardEffects(card) {
    if (!card || !card._magicCardHandlers) return;

    const handlers = card._magicCardHandlers;

    card.removeEventListener('mouseenter', handlers.mouseenter);
    card.removeEventListener('mousemove', handlers.mousemove);
    card.removeEventListener('mouseleave', handlers.mouseleave);
    card.removeEventListener('click', handlers.click);

    delete card._magicCardHandlers;

    // Remove classes
    card.classList.remove('magic-card', 'magic-card--glow', 'magic-card--hover');

    // Remove particles
    card.querySelectorAll('.magic-particle').forEach(p => {
      if (p.parentNode) p.remove();
    });

    // Reset styles
    card.style.transform = '';
    card.style.perspective = '';
    card.style.transformStyle = '';
  }

  // Find and setup all cards
  function setupAllCards() {
    // Remove effects from existing cards
    cards.forEach(c => {
      if (c.element && c.element.parentNode) {
        removeCardEffects(c.element);
      }
    });
    cards = [];

    // Find new cards using each selector
    config.cardSelectors.forEach(selector => {
      try {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
          // Skip if already processed or inside another magic card
          if (element._magicCardHandlers) return;
          if (element.closest('.magic-card') && !element.classList.contains('magic-card')) return;

          setupCardEffects(element);
          cards.push({ element });
        });
      } catch (e) {
        console.warn('Invalid selector:', selector, e);
      }
    });

    console.log(`Magic Card: Setup ${cards.length} cards`);
  }

  // Initialize
  function init(options) {
    // Merge options
    if (options) {
      Object.keys(options).forEach(key => {
        if (config.hasOwnProperty(key)) {
          config[key] = options[key];
        }
      });
    }

    // Setup spotlight
    setupSpotlight();

    // Setup cards with delay to ensure DOM is ready
    setTimeout(() => {
      setupAllCards();

      // Try again after a longer delay (for dynamic content)
      setTimeout(setupAllCards, 500);
    }, 100);

    // Global mouse move listener
    document.addEventListener('mousemove', handleGlobalMouseMove);

    // Watch for dynamically added elements
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) { // Element node
            // Check if node itself is a card
            if (isCardElement(node) && !node._magicCardHandlers) {
              setupCardEffects(node);
              cards.push({ element: node });
            }

            // Check children
            if (node.querySelectorAll) {
              config.cardSelectors.forEach(selector => {
                try {
                  const children = node.querySelectorAll(selector);
                  children.forEach(el => {
                    // Skip if already processed or inside another card
                    if (el._magicCardHandlers) return;
                    if (el.closest('.magic-card') && !el.classList.contains('magic-card')) return;

                    setupCardEffects(el);
                    cards.push({ element: el });
                  });
                } catch (e) {
                  // Skip invalid selectors
                }
              });
            }
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    console.log('Magic Card: Initialized');
  }

  // Destroy
  function destroy() {
    // Remove spotlight
    if (spotlightElement && spotlightElement.parentNode) {
      spotlightElement.parentNode.removeChild(spotlightElement);
      spotlightElement = null;
    }

    // Remove card effects
    cards.forEach(c => {
      if (c.element && c.element.parentNode) {
        removeCardEffects(c.element);
      }
    });
    cards = [];

    // Remove event listeners
    document.removeEventListener('mousemove', handleGlobalMouseMove);

    // Remove all particles and ripples
    document.querySelectorAll('.magic-particle, .magic-ripple').forEach(el => {
      if (el.parentNode) el.remove();
    });

    console.log('Magic Card: Destroyed');
  }

  // Auto-initialize
  function autoInit() {
    // Check for disable attribute
    if (document.documentElement.getAttribute('data-magic-card') === 'false') {
      return;
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      setTimeout(init, 100);
    }
  }

  // Start
  autoInit();

  // Export to global scope
  window.MagicCard = {
    init,
    destroy,
    config,
    setupAllCards
  };
})();
