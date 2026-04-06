/**
 * Starbelt - Elliptical orbit star particles with 3D perspective.
 * Stars orbit screen center in an ellipse.
 * Bottom = near (big, bright), Top = far (small, dim).
 * Center-dense: more particles near center, fewer at edges.
 */

(function () {
  'use strict';

  var STAR_COUNT = 600;
  var BASE_ANGULAR_SPEED = 0.0002;

  var stars = [];
  var canvas, ctx;
  var raf = null;
  var tiltX = 0;
  var targetTiltX = 0;
  var TILT_RANGE = Math.PI / 12;
  var TILT_LERP = 0.06;
  var CONNECTION_DIST = 60;   // max pixel distance to draw a line
  var CONNECTION_DEPTH = 0.7; // only connect particles with depth above this

  // Generate orbit radius with center-weighted distribution
  // Uses power distribution: most particles near center, few at edges
  function randomOrbitMult() {
    var u = Math.random();
    return 0.3 + Math.pow(u, 2.2) * 1.1; // range ~0.3 to ~1.4, weighted toward low
  }

  function tryInit(attempts) {
    if (document.documentElement.getAttribute('data-crosshair') === 'false') return;

    var header = document.getElementById('page-header');
    var bodyWrap = document.getElementById('body-wrap');

    if (!header || !header.classList.contains('full_page')) {
      if (attempts > 0) {
        setTimeout(function () { tryInit(attempts - 1); }, 300);
      }
      return;
    }
    if (bodyWrap && bodyWrap.classList.contains('post')) return;

    canvas = document.createElement('canvas');
    canvas.id = 'starfield-canvas';
    canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:-2;';
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
    resize();

    for (var i = 0; i < STAR_COUNT; i++) {
      stars.push({
        angle: Math.random() * Math.PI * 2,
        orbitMult: randomOrbitMult(),
        speedMult: 0.4 + Math.random() * 1.2,
        twinkleSpeed: 0.8 + Math.random() * 2,
        twinkleOffset: Math.random() * Math.PI * 2,
        baseSize: 0.3 + Math.random() * 1.2
      });
    }

    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', function (e) {
      targetTiltX = ((e.clientY / window.innerHeight) - 0.5) * 2 * TILT_RANGE;
    });
    animate();
    console.log('[Starbelt] Initialized with ' + STAR_COUNT + ' stars');
  }

  function init() { tryInit(10); }

  function resize() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function animate() {
    raf = requestAnimationFrame(animate);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Lerp tilt toward target (same speed as GLB model)
    tiltX += (targetTiltX - tiltX) * TILT_LERP;

    var cx = canvas.width / 2;
    var cy = canvas.height / 2;
    var rx = Math.min(canvas.width, canvas.height) * 0.44;
    var baseRy = rx * 0.28;
    // Tilt: mouse up → ry shrinks (top-down perspective), mouse down → ry expands
    var ry = baseRy * (1 + Math.sin(tiltX) * 1.0);
    var t = Date.now() / 1000;

    // Collect visible particle positions for connection lines
    var visible = [];

    for (var i = 0; i < stars.length; i++) {
      var s = stars[i];

      s.angle += BASE_ANGULAR_SPEED * s.speedMult * 16;

      var cosA = Math.cos(s.angle);
      var sinA = Math.sin(s.angle);
      var x = cx + cosA * rx * s.orbitMult;
      var y = cy + sinA * ry * s.orbitMult;

      // Depth: Y determines upper(far)/lower(near), X amplifies at center
      var baseDepth = (sinA + 1) / 2;
      var xFactor = 1 - Math.abs(cosA);
      var xModifier = (baseDepth - 0.5) * xFactor * 0.4;
      var depth = Math.max(0, Math.min(1, baseDepth + xModifier));

      var perspective = 0.25 + depth * 0.75;
      var radius = s.baseSize * perspective;

      var twinkle = 0.6 + 0.4 * Math.sin(t * s.twinkleSpeed + s.twinkleOffset);
      var alpha = (0.08 + depth * 0.85) * twinkle * perspective;

      var r = Math.floor(170 + depth * 85);
      var g = Math.floor(195 + depth * 60);
      var b = 255;

      if (radius < 0.15) continue;

      // Larger glow: radius * 5
      var glowR = radius * 5;
      var grad = ctx.createRadialGradient(x, y, 0, x, y, glowR);
      grad.addColorStop(0, 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')');
      grad.addColorStop(0.25, 'rgba(' + r + ',' + g + ',' + b + ',' + (alpha * 0.4) + ')');
      grad.addColorStop(0.6, 'rgba(' + r + ',' + g + ',' + b + ',' + (alpha * 0.1) + ')');
      grad.addColorStop(1, 'rgba(' + r + ',' + g + ',' + b + ',0)');

      ctx.beginPath();
      ctx.arc(x, y, glowR, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      if (depth > 0.3) {
        ctx.beginPath();
        ctx.arc(x, y, radius * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,' + (alpha * 0.7) + ')';
        ctx.fill();
      }

      // Collect for connection lines (only near/bright particles)
      if (depth > CONNECTION_DEPTH) {
        visible.push({ x: x, y: y, depth: depth, alpha: alpha });
      }
    }

    // Draw connection lines using spatial grid (avoid O(n²))
    var CELL = CONNECTION_DIST;
    var grid = {};
    for (var a = 0; a < visible.length; a++) {
      var gx = Math.floor(visible[a].x / CELL);
      var gy = Math.floor(visible[a].y / CELL);
      var key = gx + ',' + gy;
      if (!grid[key]) grid[key] = [];
      grid[key].push(visible[a]);
    }
    ctx.lineWidth = 0.5;
    ctx.beginPath(); // batch all lines into one path
    for (var key in grid) {
      var cell = grid[key];
      var parts = key.split(',');
      var gx = parseInt(parts[0]);
      var gy = parseInt(parts[1]);
      // Check this cell + 4 neighbors (right, bottom-left, bottom, bottom-right)
      var neighbors = [
        key,
        (gx + 1) + ',' + gy,
        (gx - 1) + ',' + (gy + 1),
        gx + ',' + (gy + 1),
        (gx + 1) + ',' + (gy + 1)
      ];
      for (var ni = 0; ni < neighbors.length; ni++) {
        var nkey = neighbors[ni];
        if (!grid[nkey]) continue;
        var ncell = grid[nkey];
        var startJ = (nkey === key) ? 0 : 0;
        for (var ci = 0; ci < cell.length; ci++) {
          var pa = cell[ci];
          for (var cj = startJ; cj < ncell.length; cj++) {
            var pb = (nkey === key) ? ncell[cj + 1] : ncell[cj];
            if (!pb) continue;
            var dx = pa.x - pb.x;
            var dy = pa.y - pb.y;
            var dist = dx * dx + dy * dy; // skip sqrt, compare squared
            var maxDist2 = CONNECTION_DIST * CONNECTION_DIST;
            if (dist < maxDist2) {
              // For batch path, use a fixed alpha (average depth)
              ctx.strokeStyle = 'rgba(150,200,255,0.06)';
              ctx.moveTo(pa.x, pa.y);
              ctx.lineTo(pb.x, pb.y);
            }
          }
          if (nkey === key) continue; // skip self-neighbor pairs already done
        }
      }
    }
    ctx.stroke();
  }

  function destroy() {
    if (raf) cancelAnimationFrame(raf);
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
    window.removeEventListener('resize', resize);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.VaporwaveStarfield = { init: init, destroy: destroy };
})();
