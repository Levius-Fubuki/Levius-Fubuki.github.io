/**
 * Comet Tail Cursor - Multi-state cursor with particle trail.
 * States: default, clickable (cyan+big), text (small+dim), loading (spinning ring), dragging (stretched).
 */

(function () {
  'use strict';

  var TRAIL_LENGTH = 40;
  var LERP = 0.25;
  var DOT_RADIUS = 2.5;
  var particles = [];

  // Cursor states: 'default', 'clickable', 'text', 'loading', 'dragging'
  var state = 'default';
  var stateLerp = { r: 255, g: 255, b: 255, scale: 1, opacity: 1 };

  // Target colors per state
  var STATE_TARGETS = {
    'default':  { r: 255, g: 255, b: 255, scale: 1,   opacity: 1 },
    'clickable': { r: 0,   g: 230, b: 255, scale: 1.6, opacity: 1 },
    'text':     { r: 200, g: 210, b: 230, scale: 0.5, opacity: 0.6 },
    'loading':  { r: 255, g: 180, b: 50,  scale: 1.2, opacity: 1 },
    'dragging': { r: 255, g: 220, b: 100, scale: 1.3, opacity: 0.9 }
  };

  var STATE_LERP = 0.08;

  // Hide system cursor (only on pages that run the comet effect)
  function hideSystemCursor() {
    var s = document.createElement('style');
    s.id = 'comet-cursor-hide';
    s.textContent = '* { cursor: none !important; }';
    (document.head || document.documentElement).appendChild(s);
  }

  var mx = window.innerWidth / 2, my = window.innerHeight / 2;
  var cx = mx, cy = my;
  var prevCx = cx, prevCy = cy;
  var raf = null;
  var canvas, ctx;
  var loadingAngle = 0;
  var isDragging = false;
  var dragOffsetX = 0, dragOffsetY = 0;

  function lerp(a, b, t) { return a + (b - a) * t; }

  function setState(newState) {
    if (state !== newState) {
      state = newState;
    }
  }

  function updateStateLerp() {
    var target = STATE_TARGETS[state] || STATE_TARGETS['default'];
    stateLerp.r = lerp(stateLerp.r, target.r, STATE_LERP);
    stateLerp.g = lerp(stateLerp.g, target.g, STATE_LERP);
    stateLerp.b = lerp(stateLerp.b, target.b, STATE_LERP);
    stateLerp.scale = lerp(stateLerp.scale, target.scale, STATE_LERP);
    stateLerp.opacity = lerp(stateLerp.opacity, target.opacity, STATE_LERP);
  }

  function rgbStr(alpha) {
    return 'rgba(' + Math.round(stateLerp.r) + ',' + Math.round(stateLerp.g) + ',' + Math.round(stateLerp.b) + ',' + alpha + ')';
  }

  function setupListeners() {
    // Clickable: links, buttons, inputs
    document.addEventListener('mouseover', function (e) {
      if (isDragging) return;
      if (e.target.closest('a, button, [onclick], [role="button"], summary')) {
        setState('clickable');
      } else if (e.target.closest('input[type="text"], input[type="search"], textarea, [contenteditable="true"]')) {
        setState('text');
      }
    });
    document.addEventListener('mouseout', function (e) {
      if (e.target.closest('a, button, [onclick], [role="button"], summary, input[type="text"], input[type="search"], textarea, [contenteditable="true"]')) {
        if (!isDragging) setState('default');
      }
    });

    // Dragging
    document.addEventListener('mousedown', function (e) {
      if (e.target.closest('a, button, [onclick], [role="button"]')) return;
      isDragging = true;
      setState('dragging');
    });
    document.addEventListener('mouseup', function () {
      if (isDragging) {
        isDragging = false;
        setState('default');
      }
    });

    // Page loading state
    window.addEventListener('beforeunload', function () { setState('loading'); });
    var origOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function () {
      setState('loading');
      this.addEventListener('load', function () {
        if (state === 'loading') setState('default');
      });
      this.addEventListener('error', function () {
        if (state === 'loading') setState('default');
      });
      return origOpen.apply(this, arguments);
    };
  }

  function start() {
    if (document.documentElement.getAttribute('data-crosshair') === 'false') return;

    // On article pages, disable comet cursor entirely — use system cursor
    var bodyWrap = document.getElementById('body-wrap');
    if (bodyWrap && bodyWrap.classList.contains('post')) return;

    hideSystemCursor();

    canvas = document.createElement('canvas');
    canvas.id = 'comet-canvas';
    canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;';
    document.body.appendChild(canvas);

    ctx = canvas.getContext('2d');
    resize();

    window.addEventListener('mousemove', function (e) {
      mx = e.clientX;
      my = e.clientY;
    });
    window.addEventListener('resize', resize);

    setupListeners();
    animate();
    console.log('[Crosshair] Comet cursor initialized (multi-state)');
  }

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function animate() {
    raf = requestAnimationFrame(animate);

    prevCx = cx;
    prevCy = cy;
    cx = lerp(cx, mx, LERP);
    cy = lerp(cy, my, LERP);

    updateStateLerp();

    // Dragging: offset the visual position from actual to create stretch
    var drawX = cx, drawY = cy;
    if (isDragging) {
      var velX = cx - prevCx;
      var velY = cy - prevCy;
      drawX = cx - velX * 8;
      drawY = cy - velY * 8;
    }

    particles.push({ x: drawX, y: drawY, life: 1 });
    if (particles.length > TRAIL_LENGTH) {
      particles.splice(0, particles.length - TRAIL_LENGTH);
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw trail
    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.life -= 1 / TRAIL_LENGTH;
      if (p.life <= 0) continue;

      var alpha = p.life * 0.7 * stateLerp.opacity;
      var radius = DOT_RADIUS * p.life * stateLerp.scale;
      if (radius < 0.3) continue;

      var grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 2.5);
      grad.addColorStop(0, rgbStr(alpha));
      grad.addColorStop(0.4, rgbStr(alpha * 0.4));
      grad.addColorStop(1, rgbStr(0));

      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    drawHead(drawX, drawY);
  }

  function drawHead(drawX, drawY) {
    // Draw head
    var breath = 0.7 + Math.sin(Date.now() / 400) * 0.65;
    var sc = stateLerp.scale;
    var headR = DOT_RADIUS * (2.5 + breath * 0.6) * sc;

    if (state === 'loading') {
      // Loading: rotating ring around head
      loadingAngle += 0.08;
      var ringR = DOT_RADIUS * 5 * sc;

      ctx.save();
      ctx.translate(drawX, drawY);
      ctx.rotate(loadingAngle);
      ctx.beginPath();
      ctx.arc(0, 0, ringR, 0, Math.PI * 1.2);
      ctx.strokeStyle = rgbStr(0.9);
      ctx.lineWidth = 2 * sc;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.restore();

      // Pulsing head
      var pulse = 0.6 + Math.sin(Date.now() / 200) * 0.4;
      headR = DOT_RADIUS * (2 + pulse) * sc;
    }

    var headGrad = ctx.createRadialGradient(drawX, drawY, 0, drawX, drawY, headR);
    headGrad.addColorStop(0, rgbStr(0.9 + breath * 0.1));
    headGrad.addColorStop(0.3, rgbStr(0.5 * breath + 0.2));
    headGrad.addColorStop(0.7, rgbStr(0.2 * breath));
    headGrad.addColorStop(1, rgbStr(0));

    ctx.beginPath();
    ctx.arc(drawX, drawY, headR, 0, Math.PI * 2);
    ctx.fillStyle = headGrad;
    ctx.fill();

    // Core dot
    var coreR = DOT_RADIUS * (0.5 + breath * 0.2) * sc;
    if (state === 'text') coreR *= 0.4; // tiny for text
    ctx.beginPath();
    ctx.arc(drawX, drawY, coreR, 0, Math.PI * 2);
    ctx.fillStyle = rgbStr(stateLerp.opacity);
    ctx.fill();

    // Draw drag stretch line between offset position and actual cursor
    if (isDragging) {
      ctx.beginPath();
      ctx.moveTo(drawX, drawY);
      ctx.lineTo(cx, cy);
      ctx.strokeStyle = rgbStr(0.4);
      ctx.lineWidth = 2 * sc;
      ctx.lineCap = 'round';
      ctx.stroke();
    }
  }

  function destroy() {
    if (raf) cancelAnimationFrame(raf);
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
    var s = document.getElementById('comet-cursor-hide');
    if (s) s.remove();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  window.Crosshair = { init: start, destroy: destroy };
})();
