/**
 * 3D GLB Background - Renders a GLTF model as fixed background.
 * Model rotation follows mouse position: center=default, edges=tilt.
 * Only active on homepage. Article pages are excluded (cool blue theme).
 */

(function () {
  'use strict';

  var CONFIG = {
    glbPath: '/img/bg.glb',
    rotYRange: Math.PI / 2, // Horizontal: left/right edge → 90°
    rotXRange: Math.PI / 12, // Vertical: top/bottom edge → 15° (slight tilt)
    lerpSpeed: 0.06,
    bgColor: 0x0a0a0a,
  };

  var scene, camera, renderer, model, animationId;
  var currentRotX = 0, currentRotY = 0;
  var targetRotX = 0, targetRotY = 0;
  var isActive = false;
  var isLoaded = false;

  function isArticlePage() {
    const bodyWrap = document.getElementById('body-wrap');
    return !!(bodyWrap && bodyWrap.classList.contains('post'));
  }

  function isHomePage() {
    const header = document.getElementById('page-header');
    return !!(header && header.classList.contains('full_page'));
  }

  function shouldActivate() {
    if (isArticlePage()) return false;
    return isHomePage();
  }

  function createCanvas() {
    const existing = document.getElementById('glb-background-canvas');
    if (existing) existing.remove();

    const canvas = document.createElement('canvas');
    canvas.id = 'glb-background-canvas';
    canvas.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: -3;
      pointer-events: none;
    `;
    document.body.appendChild(canvas);
    return canvas;
  }

  function initThree(canvas) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.bgColor);

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
  }

  function loadModel() {
    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath('/js/vendor/draco/');
    dracoLoader.setDecoderConfig({ type: 'js' });

    const loader = new THREE.GLTFLoader();
    loader.setDRACOLoader(dracoLoader);
    loader.load(
      CONFIG.glbPath,
      function (gltf) {
        model = gltf.scene;

        // Fit model to camera view
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        const fitDistance = maxDim / (2 * Math.tan(fov / 2));

        model.position.sub(center);
        camera.position.set(0, 0, fitDistance * 1.2);

        // Add ambient + directional light
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 5, 5);
        scene.add(dirLight);

        scene.add(model);
        isLoaded = true;

        scene.add(model);
        isLoaded = true;

        animate();

        window.dispatchEvent(new CustomEvent('glb-background-loaded'));
      },
      function (progress) {
        if (progress.total > 0) {
          var pct = Math.round((progress.loaded / progress.total) * 100);
          console.log('[GLB Background] Loading:', pct + '%');
        }
      },
      function (error) {
        console.error('[GLB Background] Failed to load:', error);
        cleanup();
      }
    );
  }

  function onMouseMove(e) {
    if (!isLoaded) return;
    // Normalize mouse to -1 ~ 1 (center = 0)
    targetRotY = ((e.clientX / window.innerWidth) - 0.5) * 2 * CONFIG.rotYRange;
    targetRotX = ((e.clientY / window.innerHeight) - 0.5) * 2 * CONFIG.rotXRange;
  }

  function onResize() {
    if (!renderer || !camera) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function animate() {
    if (!isActive) return;

    animationId = requestAnimationFrame(animate);

    // Smooth lerp rotation toward mouse-driven target
    currentRotX += (targetRotX - currentRotX) * CONFIG.lerpSpeed;
    currentRotY += (targetRotY - currentRotY) * CONFIG.lerpSpeed;

    if (model) {
      model.rotation.y = currentRotY;
      model.rotation.x = currentRotX;
    }

    renderer.render(scene, camera);
  }

  function cleanup() {
    isActive = false;
    if (animationId) cancelAnimationFrame(animationId);
    var canvas = document.getElementById('glb-background-canvas');
    if (canvas) canvas.remove();
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('resize', onResize);
  }

  function handleVisibility() {
    if (document.hidden) {
      isActive = false;
      if (animationId) cancelAnimationFrame(animationId);
    } else if (isLoaded && shouldActivate()) {
      isActive = true;
      animate();
    }
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('Failed to load: ' + src)); };
      document.head.appendChild(s);
    });
  }

  function loadThreeJSCDN() {
    if (window.THREE && THREE.GLTFLoader) {
      return Promise.resolve();
    }

    return loadScript('/js/vendor/three.min.js')
      .then(function () {
        return loadScript('/js/vendor/DRACOLoader.js');
      })
      .then(function () {
        return loadScript('/js/vendor/GLTFLoader.js');
      });
  }

  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    if (isArticlePage()) {
      console.log('[GLB Background] Skipped: article page');
      return;
    }

    if (!shouldActivate()) {
      console.log('[GLB Background] Skipped: not homepage');
      return;
    }

    isActive = true;

    loadThreeJSCDN()
      .then(function () {
        var canvas = createCanvas();
        initThree(canvas);
        loadModel();

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('resize', onResize);
        document.addEventListener('visibilitychange', handleVisibility);

        console.log('[GLB Background] Initialized (mouse-follow)');
      })
      .catch(function (err) {
        console.error('[GLB Background] Failed to load Three.js:', err);
        cleanup();
      });
  }

  init();

  window.VaporwaveGLB = {
    config: CONFIG,
    cleanup: cleanup,
    init: init,
    isLoaded: function () { return isLoaded; },
    isActive: function () { return isActive; },
  };
})();
