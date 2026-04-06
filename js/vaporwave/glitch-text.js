/**
 * Glitch Text Effect
 * Cyberpunk glitch animation on the site title (#site-title).
 * RGB channel split, random displacement, scanline flicker.
 */

(function () {
  'use strict';

  var INTERVAL = 3000;     // Glitch every 3s
  var GLITCH_DURATION = 200; // Each glitch lasts 200ms
  var MAX_OFFSET = 4;       // Max pixel displacement

  var styleEl = null;

  function injectStyles() {
    styleEl = document.createElement('style');
    styleEl.id = 'glitch-text-styles';
    styleEl.textContent = [
      // Glitch container
      '#site-title { position: relative; display: inline-block; width: 100%; text-align: center; }',

      // Glitch pseudo layers (before = red, after = cyan)
      '#site-title.glitch::before,',
      '#site-title.glitch::after {',
      '  content: attr(data-text);',
      '  position: absolute;',
      '  top: 0; left: 0;',
      '  width: 100%; height: 100%;',
      '  overflow: hidden;',
      '}',

      '#site-title.glitch::before {',
      '  color: #FF0040;',
      '  z-index: -1;',
      '  animation: glitchClip1 2s infinite linear alternate-reverse;',
      '}',

      '#site-title.glitch::after {',
      '  color: #00FFFF;',
      '  z-index: -2;',
      '  animation: glitchClip2 2s infinite linear alternate-reverse;',
      '}',

      // Clip animations for scanline displacement
      '@keyframes glitchClip1 {',
      '  0% { clip-path: inset(40% 0 61% 0); transform: translate(-2px, -1px); }',
      '  20% { clip-path: inset(92% 0 1% 0); transform: translate(1px, 2px); }',
      '  40% { clip-path: inset(43% 0 1% 0); transform: translate(-1px, 0); }',
      '  60% { clip-path: inset(25% 0 58% 0); transform: translate(2px, -1px); }',
      '  80% { clip-path: inset(54% 0 7% 0); transform: translate(-2px, 1px); }',
      '  100% { clip-path: inset(58% 0 43% 0); transform: translate(0); }',
      '}',

      '@keyframes glitchClip2 {',
      '  0% { clip-path: inset(65% 0 13% 0); transform: translate(2px, 1px); }',
      '  20% { clip-path: inset(15% 0 62% 0); transform: translate(-1px, -2px); }',
      '  40% { clip-path: inset(78% 0 1% 0); transform: translate(0, 1px); }',
      '  60% { clip-path: inset(2% 0 85% 0); transform: translate(1px, 0); }',
      '  80% { clip-path: inset(33% 0 45% 0); transform: translate(-2px, -1px); }',
      '  100% { clip-path: inset(10% 0 71% 0); transform: translate(0); }',
      '}',

      // Instant glitch burst
      '#site-title.glitch-burst {',
      '  animation: glitchBurst 0.15s steps(2) 3;',
      '}',

      '@keyframes glitchBurst {',
      '  0% { text-shadow: 2px 0 #FF0040, -2px 0 #00FFFF; transform: translate(1px, -1px); }',
      '  25% { text-shadow: -2px 0 #FF0040, 2px 0 #00FFFF; transform: translate(-1px, 1px); }',
      '  50% { text-shadow: 1px 1px #FF0040, -1px -1px #00FFFF; transform: translate(0); }',
      '  75% { text-shadow: -1px 0 #FF0040, 1px 0 #00FFFF; transform: translate(2px, 0); }',
      '  100% { text-shadow: none; transform: translate(0); }',
      '}',

      // Mobile: disable glitch
      '@media (max-width: 768px) {',
      '  #site-title.glitch::before, #site-title.glitch::after { display: none !important; }',
      '  #site-title.glitch-burst { animation: none !important; }',
      '}'
    ].join('\n');

    (document.head || document.documentElement).appendChild(styleEl);
  }

  function applyGlitch() {
    var title = document.getElementById('site-title');
    if (!title) {
      // Retry until title exists (pjax navigation)
      setTimeout(applyGlitch, 500);
      return;
    }

    // Set data-text for pseudo elements
    title.setAttribute('data-text', title.textContent);

    // Activate persistent subtle glitch
    title.classList.add('glitch');

    // Periodic burst glitch
    setInterval(function () {
      title.classList.remove('glitch-burst');
      // Force reflow
      void title.offsetWidth;
      title.classList.add('glitch-burst');
      setTimeout(function () {
        title.classList.remove('glitch-burst');
      }, GLITCH_DURATION);
    }, INTERVAL);
  }

  function init() {
    if (document.documentElement.getAttribute('data-crosshair') === 'false') return;

    injectStyles();

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () {
        setTimeout(applyGlitch, 300);
      });
    } else {
      setTimeout(applyGlitch, 300);
    }

    // Reapply after pjax navigation
    document.addEventListener('pjax:complete', function () {
      setTimeout(applyGlitch, 300);
    });
  }

  function destroy() {
    if (styleEl) styleEl.remove();
    var title = document.getElementById('site-title');
    if (title) {
      title.classList.remove('glitch', 'glitch-burst');
      title.removeAttribute('data-text');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.VaporwaveGlitch = { init: init, destroy: destroy };
})();
