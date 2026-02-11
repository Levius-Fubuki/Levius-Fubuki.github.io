/**
 * Vaporwave Holographic 3D Card Effect
 * Applies full holographic effects to blog cards with mouse tracking
 *
 * Target cards: .recent-post-item, .archive-item, .category-item, .tag-item, .post-copyright, .card-widget
 */

(function() {
  'use strict';

  // ==============================================
  // Configuration
  // ==============================================
  const CONFIG = {
    // Target card selectors - matched to Butterfly theme's actual HTML structure
    // EXCLUDED: .aside-list-item (sidebar "Recent Posts" items), .post-copyright (copyright notice)
    cardSelectors: [
      '.recent-post-item',           // Homepage recent post cards
      '.article-sort-item',          // Archive list items (not .archive-item)
      '.category-list-item',         // Category list items (if exists)
      '.tag-list-item',              // Tag list items (if exists)
      '#aside-content .card-widget'  // Sidebar card widgets (use higher specificity)
    ],
    // Animation timing
    smoothDuration: 600,
    initialDuration: 1500,
    // Proximity detection range (pixels) - card will tilt when mouse is within this distance
    proximityRange: 150,
    // Reduce motion for accessibility
    prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches
  };

  // ==============================================
  // Utility Functions
  // ==============================================
  const clamp = (value, min = 0, max = 100) => Math.min(Math.max(value, min), max);

  const round = (value, precision = 3) => parseFloat(value.toFixed(precision));

  const adjust = (value, fromMin, fromMax, toMin, toMax) =>
    round(toMin + ((toMax - toMin) * (value - fromMin)) / (fromMax - fromMin));

  // Easing function for smooth animation
  const easeInOutCubic = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);

  // ==============================================
  // Core Effect Functions
  // ==============================================

  /**
   * Update card CSS variables based on pointer position
   */
  const updateCardTransform = (offsetX, offsetY, card) => {
    const width = card.clientWidth;
    const height = card.clientHeight;

    // Calculate percentage position
    const percentX = clamp((100 / width) * offsetX);
    const percentY = clamp((100 / height) * offsetY);

    // Calculate distance from center (-50 to 50)
    const centerX = percentX - 50;
    const centerY = percentY - 50;

    // Set CSS custom properties
    card.style.setProperty('--holo-pointer-x', `${percentX}%`);
    card.style.setProperty('--holo-pointer-y', `${percentY}%`);
    card.style.setProperty('--holo-background-x', `${adjust(percentX, 0, 100, 35, 65)}%`);
    card.style.setProperty('--holo-background-y', `${adjust(percentY, 0, 100, 35, 65)}%`);
    card.style.setProperty('--holo-pointer-from-center',
      `${clamp(Math.hypot(percentY - 50, percentX - 50) / 50, 0, 1)}`);
    card.style.setProperty('--holo-pointer-from-top', `${percentY / 100}`);
    card.style.setProperty('--holo-pointer-from-left', `${percentX / 100}`);

    // Check if current card is an archive article list item for reduced tilt effect
    const isArchiveArticleItem = card.classList.contains('article-sort-item');

    // Calculate 3D rotation - mouse position corner tilts AWAY (outward tilt)
    // Goal: mouse at corner = that corner tilts away/down (opposite corner comes forward)
    // Archive article list items use reduced tilt for subtler effect
    const tiltDividerX = isArchiveArticleItem ? 10 : 6;
    const tiltDividerY = isArchiveArticleItem ? 12 : 8;
    const rotateX = round(centerY / tiltDividerX);   // Controls up/down tilt
    const rotateY = round(centerX / tiltDividerY);   // Controls left/right tilt

    card.style.setProperty('--holo-rotate-x', `${rotateY}deg`);  // SWAPPED: X value → --rotate-x → CSS rotateY
    card.style.setProperty('--holo-rotate-y', `${rotateX}deg`);  // SWAPPED: Y value → --rotate-y → CSS rotateX

    // Debug log
    if (card._holoDebug) {
      console.log(`[${card.className}] X:${percentX.toFixed(1)}% Y:${percentY.toFixed(1)}% rotateX:${rotateX}° rotateY:${rotateY}°`);
    }
  };

  /**
   * Create smooth return-to-center animation
   */
  const createSmoothAnimation = (duration, startX, startY, card) => {
    const startTime = performance.now();
    const targetX = card.clientWidth / 2;
    const targetY = card.clientHeight / 2;

    const animationLoop = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = clamp(elapsed / duration);
      const easedProgress = easeInOutCubic(progress);

      const currentX = adjust(easedProgress, 0, 1, startX, targetX);
      const currentY = adjust(easedProgress, 0, 1, startY, targetY);

      updateCardTransform(currentX, currentY, card);

      if (progress < 1) {
        card._holoRafId = requestAnimationFrame(animationLoop);
      } else {
        card.classList.remove('holo-active');
        card.style.setProperty('--holo-card-opacity', '0');
        // Reset rotation to center
        card.style.setProperty('--holo-rotate-x', '0deg');
        card.style.setProperty('--holo-rotate-y', '0deg');
      }
    };

    card._holoRafId = requestAnimationFrame(animationLoop);
  };

  /**
   * Cancel any running animation on a card
   */
  const cancelAnimation = (card) => {
    if (card._holoRafId) {
      cancelAnimationFrame(card._holoRafId);
      card._holoRafId = null;
    }
  };

  // ==============================================
  // Event Handlers
  // ==============================================

  /**
   * Check if mouse is within proximity range of card
   */
  const isMouseInProximity = (mouseX, mouseY, card) => {
    const rect = card.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const distance = Math.hypot(mouseX - centerX, mouseY - centerY);
    const maxDistance = Math.max(rect.width, rect.height) / 2 + CONFIG.proximityRange;
    return distance <= maxDistance;
  };

  /**
   * Get projected mouse position on card (clamped to card bounds)
   */
  const getProjectedPosition = (mouseX, mouseY, card) => {
    const rect = card.getBoundingClientRect();
    return {
      x: clamp(mouseX - rect.left, 0, rect.width),
      y: clamp(mouseY - rect.top, 0, rect.height)
    };
  };

  /**
   * Handle pointer move on card
   */
  const handlePointerMove = (event, card) => {
    if (CONFIG.prefersReducedMotion) return;

    const rect = card.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    updateCardTransform(x, y, card);
  };

  /**
   * Handle pointer enter on card
   */
  const handlePointerEnter = (card) => {
    if (CONFIG.prefersReducedMotion) return;

    cancelAnimation(card);
    card.classList.add('holo-active');
    card.style.setProperty('--holo-card-opacity', '1');

    // Add glare element if not exists
    if (!card.querySelector('.holo-glare')) {
      const glare = document.createElement('div');
      glare.className = 'holo-glare';
      card.appendChild(glare);
    }
  };

  /**
   * Handle pointer leave on card
   */
  const handlePointerLeave = (event, card) => {
    if (CONFIG.prefersReducedMotion) return;

    const rect = card.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    createSmoothAnimation(CONFIG.smoothDuration, x, y, card);
    card.classList.remove('holo-active');
  };

  // ==============================================
  // Card Initialization
  // ==============================================

  /**
   * Initialize event listeners for a single card
   */
  const initCard = (card) => {
    // Skip if already initialized
    if (card._holoInitialized) return;
    card._holoInitialized = true;

    // Ensure card has position for absolute positioning of glare
    if (getComputedStyle(card).position === 'static') {
      card.style.position = 'relative';
    }

    // Mouse events - use mousemove for smooth tracking
    card.addEventListener('mousemove', (e) => {
      if (CONFIG.prefersReducedMotion) return;
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if (!card.classList.contains('holo-active')) {
        handlePointerEnter(card);
      }
      updateCardTransform(x, y, card);
    });

    // Touch events for mobile
    card.addEventListener('touchstart', (e) => {
      if (CONFIG.prefersReducedMotion) return;
      const touch = e.touches[0];
      const rect = card.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;
      handlePointerEnter(card);
      updateCardTransform(x, y, card);
    }, { passive: true });

    card.addEventListener('touchmove', (e) => {
      if (CONFIG.prefersReducedMotion) return;
      const touch = e.touches[0];
      handlePointerMove(touch, card);
    }, { passive: true });

    card.addEventListener('touchend', (e) => {
      if (CONFIG.prefersReducedMotion) return;
      const rect = card.getBoundingClientRect();
      // Use center of card as end position
      const x = card.clientWidth / 2;
      const y = card.clientHeight / 2;
      createSmoothAnimation(CONFIG.smoothDuration, x, y, card);
      card.classList.remove('holo-active');
    });
  };

  /**
   * Initialize all holographic cards on the page
   */
  const initAllCards = () => {
    CONFIG.cardSelectors.forEach(selector => {
      const cards = document.querySelectorAll(selector);
      console.log(`[HoloCard] Found ${cards.length} cards for selector: ${selector}`);
      cards.forEach(card => initCard(card));
    });

    // Setup global mouse tracking for proximity detection
    document.addEventListener('mousemove', (e) => {
      if (CONFIG.prefersReducedMotion) return;

      const allCards = document.querySelectorAll(CONFIG.cardSelectors.join(','));
      allCards.forEach(card => {
        if (isMouseInProximity(e.clientX, e.clientY, card)) {
          if (!card.classList.contains('holo-active')) {
            handlePointerEnter(card);
          }
          const pos = getProjectedPosition(e.clientX, e.clientY, card);
          updateCardTransform(pos.x, pos.y, card);
        } else if (card.classList.contains('holo-active')) {
          // Mouse left proximity range, return to center
          createSmoothAnimation(CONFIG.smoothDuration, card.clientWidth / 2, card.clientHeight / 2, card);
          card.classList.remove('holo-active');
        }
      });
    });
  };

  // ==============================================
  // Initialize on DOM Ready
  // ==============================================
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllCards);
  } else {
    initAllCards();
  }

  // Re-initialize when page content changes (for SPA navigation)
  const observePageChanges = () => {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.addedNodes.length > 0) {
          // Check if any added nodes contain cards
          mutation.addedNodes.forEach(node => {
            if (node.nodeType === 1) { // Element node
              CONFIG.cardSelectors.forEach(selector => {
                if (node.matches && node.matches(selector)) {
                  initCard(node);
                }
                const cards = node.querySelectorAll && node.querySelectorAll(selector);
                if (cards) {
                  cards.forEach(card => initCard(card));
                }
              });
            }
          });
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  };

  observePageChanges();

  // ==============================================
  // Cleanup on page unload
  // ==============================================
  window.addEventListener('beforeunload', () => {
    const cards = document.querySelectorAll(CONFIG.cardSelectors.join(','));
    cards.forEach(card => {
      cancelAnimation(card);
    });
  });

  // Export for external access
  window.VaporwaveHolographicCard = {
    init: initAllCards,
    update: updateCardTransform,
    CONFIG
  };

  console.log('[HoloCard] Initialized with selectors:', CONFIG.cardSelectors);

})();
