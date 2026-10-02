(() => {
  "use strict";
  const scene = document.querySelector(".hero-scene");
  const canvas = document.querySelector(".ambient-field");
  const toggle = document.querySelector(".motion-toggle");
  if (!scene || !canvas || !toggle) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const root = document.documentElement;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = matchMedia("(hover: hover) and (pointer: fine)");
  const dialog = document.querySelector("#search-dialog");
  const title = document.querySelector("#hero-title .text-ink");
  const animations = new Set();
  let enabled = true;
  try { enabled = localStorage.getItem("levius-ambient-motion") !== "off"; } catch {}
  let inView = true, running = false, frame = 0, lastTime = 0, elapsed = 0;
  let width = 0, height = 0, pointerX = 0, pointerY = 0, driftX = 0, driftY = 0;
  let decodeFrame = 0;
  document.querySelector(".scene-status").hidden = false;

  function animate(element, delay = 0, distance = 14) {
    if (!element || reduced.matches || !enabled) return;
    const animation = element.animate([
      { opacity: 0, transform: `translateY(${distance}px)` },
      { opacity: 1, transform: "translateY(0)" },
    ], { duration: 680, delay, easing: "cubic-bezier(.2,.75,.25,1)", fill: "backwards" });
    animations.add(animation);
    animation.onfinish = () => animations.delete(animation);
  }

  // The accessible heading is static; only its decorative copy is decoded.
  function decodeTitle() {
    if (!title || reduced.matches || !enabled) return;
    const word = "LEVIUS", glyphs = "01_/:+";
    const started = performance.now();
    let previousStep = -1;
    function step(now) {
      const age = now - started;
      const tick = Math.floor(age / 70);
      if (tick !== previousStep) {
        previousStep = tick;
        const settled = Math.floor(Math.max(0, age - 140) / 100);
        title.textContent = [...word].map((letter, i) => i < settled ? letter : glyphs[(tick + i * 3) % glyphs.length]).join("");
      }
      if (age < 850 && running) decodeFrame = requestAnimationFrame(step);
      else { title.textContent = word; decodeFrame = 0; }
    }
    decodeFrame = requestAnimationFrame(step);
  }

  function stopEntrance() {
    cancelAnimationFrame(decodeFrame);
    decodeFrame = 0;
    if (title) title.textContent = "LEVIUS";
    animations.forEach(animation => animation.cancel());
    animations.clear();
  }

  // A single field of drifting ash and low mist spans the entire medieval scene.
  function draw() {
    ctx.clearRect(0, 0, width, height);
    const t = elapsed / 1000;
    for (let i = 0; i < 46; i++) {
      const x = ((i * 137.51 + Math.sin(t * .16 + i) * 16 + driftX * .3) % width + width) % width;
      const y = ((i * 79.37 - t * (3 + i % 5) + driftY * .2) % height + height) % height;
      const alpha = .08 + .15 * (.5 + .5 * Math.sin(t * .5 + i));
      ctx.fillStyle = `rgba(222,222,222,${alpha})`;
      ctx.beginPath();ctx.arc(x, y, i % 7 === 0 ? 1.2 : .6, 0, Math.PI * 2);ctx.fill();
    }
    for (let i = 0; i < 3; i++) {
      const x = width * (i * .42 + .08) + Math.sin(t * .075 + i) * 70;
      const y = height * (.84 + Math.sin(t * .12 + i) * .05);
      ctx.save();ctx.translate(x, y);ctx.scale(1, .22);
      const radius = width * .52;
      const fog = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
      fog.addColorStop(0, 'rgba(210,210,210,.055)');fog.addColorStop(1, 'rgba(210,210,210,0)');
      ctx.fillStyle = fog;ctx.fillRect(-radius, -radius, radius * 2, radius * 2);ctx.restore();
    }
  }

  function tick(now) {
    if (!running) return;
    frame = requestAnimationFrame(tick);
    if (now - lastTime < 32) return;
    elapsed += Math.min(now - lastTime, 64);
    lastTime = now;
    driftX += (pointerX - driftX) * .055;
    driftY += (pointerY - driftY) * .055;
    draw();
  }

  function sync() {
    const allowed = enabled && !reduced.matches;
    running = allowed && inView && !document.hidden && !dialog?.open;
    root.classList.toggle("motion-running", running);
    toggle.setAttribute("aria-pressed", String(allowed));
    toggle.textContent = reduced.matches ? "动效：随系统关闭" : `环境动效：${enabled ? "开" : "关"}`;
    toggle.disabled = reduced.matches;
    cancelAnimationFrame(frame);
    frame = 0;
    if (running) { lastTime = performance.now(); frame = requestAnimationFrame(tick); }
    else stopEntrance();
    draw();
  }

  function resize() {
    const bounds = scene.getBoundingClientRect();
    width = bounds.width; height = bounds.height;
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  new ResizeObserver(resize).observe(scene);
  new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; sync(); }).observe(scene);
  document.addEventListener("visibilitychange", sync);
  reduced.addEventListener("change", sync);
  if (dialog) new MutationObserver(sync).observe(dialog, { attributes: true, attributeFilter: ["open"] });
  function setEnabled(value) {
    enabled = Boolean(value);
    try { localStorage.setItem("levius-ambient-motion", enabled ? "on" : "off"); } catch {}
    sync();
  }
  window.LeviusMotion = Object.freeze({ setEnabled, getState: () => ({ enabled, reduced: reduced.matches }) });
  toggle.addEventListener("click", () => setEnabled(!enabled));
  scene.addEventListener("pointermove", event => {
    if (!running || !finePointer.matches) return;
    const bounds = scene.getBoundingClientRect();
    pointerX = ((event.clientX - bounds.left) / bounds.width - .5) * 16;
    pointerY = ((event.clientY - bounds.top) / bounds.height - .5) * 12;
  }, { passive: true });
  scene.addEventListener("pointerleave", () => { pointerX = 0; pointerY = 0; });
  resize(); sync();
  animate(document.querySelector(".site-header"), 0, 6);
  animate(document.querySelector(".hero-window"), 70, 10);
  ["#hero-title", ".hero-intro", ".hero-topics", ".hero-actions"].forEach((selector, i) => animate(document.querySelector(selector), 130 + i * 90));
  if (!window.LeviusPages) decodeTitle();
  const reveals = new IntersectionObserver(entries => {
    let order = 0;
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      if (!document.hidden) animate(entry.target, order++ * 85, 12);
      reveals.unobserve(entry.target);
    }
  }, { threshold: .08 });
  document.querySelectorAll(".terminal-masthead, .terminal-sidebar, .terminal-console, .section-heading, .entry, .site-footer").forEach(element => reveals.observe(element));
})();
