// Vaporwave Background Fix
console.log('Vaporwave background-fix.js loaded');

function setBackground() {
  console.log('setBackground called');
  const h = document.getElementById('page-header');
  console.log('header element:', h);

  if (h) {
    h.style.setProperty('background-image', 'url(/img/background.jpg)', 'important');
    h.style.setProperty('background-size', 'cover', 'important');
    h.style.setProperty('background-position', 'center', 'important');
    h.style.setProperty('z-index', '10', 'important');
    console.log('Vaporwave background applied');
    console.log('Background image:', window.getComputedStyle(h).backgroundImage);
  } else {
    console.log('header not found, retrying...');
    setTimeout(setBackground, 100);
  }
}

// Fix for article pages - Cool Blue/Cyan theme
function fixArticlePageBackground() {
  console.log('fixArticlePageBackground called');

  // Check if this is an article page
  const bodyWrap = document.getElementById('body-wrap');
  if (bodyWrap && bodyWrap.classList.contains('post')) {
    console.log('Article page detected, applying cool blue background');

    // Add class to body for CSS targeting
    document.body.classList.add('body-article-page');

    // Set body background to cool blue
    document.body.style.setProperty('background', '#081828', 'important');
    document.body.style.setProperty('background-image', 'none', 'important');

    // Set body-wrap background to cool blue
    bodyWrap.style.setProperty('background', '#081828', 'important');

    // CRITICAL: Create a style element to override scrollbar styles
    // This bypasses CSS specificity issues
    let styleEl = document.getElementById('article-page-scrollbar-styles');
    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.id = 'article-page-scrollbar-styles';
      document.head.appendChild(styleEl);
    }

    // Set scrollbar styles directly in the style element
    styleEl.textContent = `
      /* Scrollbar track */
      ::-webkit-scrollbar {
        width: 12px !important;
        height: 12px !important;
      }
      ::-webkit-scrollbar-track {
        background: #051520 !important;
      }
      /* Scrollbar thumb - cool blue gradient */
      ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #00B4DC, #00D4FF) !important;
        border-radius: 6px !important;
        border: 2px solid #051520 !important;
      }
      ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(to bottom, #00D4FF, #00FFFF) !important;
      }
    `;

    console.log('Article page background and scrollbar applied');

    // Fix TOC card background - Cool Blue theme
    fixTOCCardBackground();
  } else {
    console.log('Not an article page or body-wrap not found');
  }
}

// Fix TOC Card Background - Cool Blue theme
function fixTOCCardBackground() {
  console.log('fixTOCCardBackground called');

  // Get the TOC card element directly
  const tocCard = document.getElementById('card-toc');
  if (!tocCard) {
    console.log('TOC card not found');
    return;
  }

  // CRITICAL: Set CSS variables on body element to ensure inheritance
  document.body.style.setProperty('--card-bg', 'rgba(8, 24, 40, 0.9)', 'important');
  document.body.style.setProperty('--card-box-shadow', '0 4px 20px rgba(0, 40, 70, 0.5)', 'important');
  document.body.style.setProperty('--card-hover-box-shadow', '0 6px 25px rgba(0, 60, 90, 0.6)', 'important');

  // CRITICAL: Also set styles on the .sticky_layout parent
  const stickyLayout = tocCard.closest('.sticky_layout');
  if (stickyLayout) {
    stickyLayout.style.setProperty('background', 'transparent', 'important');
    stickyLayout.style.setProperty('--card-bg', 'rgba(8, 24, 40, 0.9)', 'important');
    console.log('.sticky_layout styles applied');
  }

  // CRITICAL: Fix the .toc element which has purple background
  const tocElement = tocCard.querySelector('.toc') || document.getElementById('toc');
  if (tocElement) {
    tocElement.style.setProperty('background', 'transparent', 'important');
    tocElement.style.setProperty('background-color', 'transparent', 'important');
    console.log('.toc element styles applied - fixed purple background');
  }

  // Apply inline styles directly to the TOC card element
  // This has the highest specificity and will override all CSS
  tocCard.style.setProperty('background', 'rgba(8, 24, 40, 0.9)', 'important');
  tocCard.style.setProperty('background-color', 'rgba(8, 24, 40, 0.9)', 'important');
  tocCard.style.setProperty('border', '1px solid rgba(0, 150, 200, 0.3)', 'important');
  tocCard.style.setProperty('border-color', 'rgba(0, 150, 200, 0.3)', 'important');
  tocCard.style.setProperty('box-shadow', '0 4px 20px rgba(0, 40, 70, 0.5)', 'important');
  tocCard.style.setProperty('transform', 'none', 'important');
  tocCard.style.setProperty('transform-style', 'flat', 'important');
  tocCard.style.setProperty('perspective', 'none', 'important');
  tocCard.style.setProperty('backdrop-filter', 'blur(10px)', 'important');

  // Also set CSS variables for completeness
  tocCard.style.setProperty('--card-bg', 'rgba(8, 24, 40, 0.9)', 'important');
  tocCard.style.setProperty('--card-box-shadow', '0 4px 20px rgba(0, 40, 70, 0.5)', 'important');
  tocCard.style.setProperty('--card-hover-box-shadow', '0 6px 25px rgba(0, 60, 90, 0.6)', 'important');

  // CRITICAL: Force override any background property that uses var()
  tocCard.style.removeProperty('background');

  // Debug: Log the computed background
  const computedBg = window.getComputedStyle(tocCard).backgroundColor;
  const computedBgImg = window.getComputedStyle(tocCard).backgroundImage;
  console.log('TOC card computed background-color:', computedBg);
  console.log('TOC card computed background-image:', computedBgImg);
  console.log('TOC card inline style background:', tocCard.style.background);
  console.log('.sticky_layout element:', stickyLayout);
  console.log('TOC card inline styles applied');

  // Create a style element for additional CSS rules
  let tocStyleEl = document.getElementById('article-page-toc-styles');
  if (!tocStyleEl) {
    tocStyleEl = document.createElement('style');
    tocStyleEl.id = 'article-page-toc-styles';
    document.head.appendChild(tocStyleEl);
  }

  // Set additional CSS rules
  tocStyleEl.textContent = `
    /* Aside content background - parent of TOC */
    #aside-content,
    #aside-content .sticky_layout {
      background: transparent !important;
    }

    /* Remove pseudo-elements for TOC card */
    #aside-content #card-toc::before,
    #card-toc::before,
    #aside-content #card-toc::after,
    #card-toc::after,
    #aside-content #card-toc:hover::before,
    #card-toc:hover::before,
    #aside-content #card-toc:hover::after,
    #card-toc:hover::after {
      display: none !important;
    }

    /* TOC Content - Fix margin and background */
    #aside-content #card-toc .toc-content,
    #card-toc .toc-content {
      background: transparent !important;
      margin: 0 -24px !important;
      width: calc(100% + 48px) !important;
    }

    /* CRITICAL: Fix .toc element purple background */
    #aside-content #card-toc .toc,
    #card-toc .toc,
    #aside-content .toc,
    .toc {
      background: transparent !important;
      background-color: transparent !important;
      border: none !important;
      border-color: transparent !important;
    }

    /* CRITICAL: Fix TOC card border - remove purple, add cool blue */
    #aside-content #card-toc,
    #card-toc {
      border: 1px solid rgba(0, 150, 200, 0.3) !important;
      border-color: rgba(0, 150, 200, 0.3) !important;
      outline: none !important;
    }

    #aside-content #card-toc:hover,
    #card-toc:hover {
      border: 1px solid rgba(0, 180, 220, 0.5) !important;
      border-color: rgba(0, 180, 220, 0.5) !important;
      outline: none !important;
    }

    /* TOC Content children - fix spacing */
    #aside-content #card-toc .toc-content > *,
    #card-toc .toc-content > * {
      margin: 0 20px !important;
    }

    /* TOC Links */
    #aside-content #card-toc .toc-content .toc-link,
    #card-toc .toc-content .toc-link,
    #aside-content #card-toc .toc-link,
    #card-toc .toc-link {
      color: #8AB8C8 !important;
      border-radius: 4px;
      transition: all 0.3s ease;
    }

    #aside-content #card-toc .toc-content .toc-link:hover,
    #card-toc .toc-content .toc-link:hover,
    #aside-content #card-toc .toc-link:hover,
    #card-toc .toc-link:hover {
      color: #00D4FF !important;
      background: rgba(0, 40, 70, 0.3) !important;
    }

    #aside-content #card-toc .toc-content .toc-link.active,
    #card-toc .toc-content .toc-link.active,
    #aside-content #card-toc .toc-link.active,
    #card-toc .toc-link.active {
      background: linear-gradient(90deg, rgba(0, 180, 220, 0.4), rgba(0, 150, 200, 0.3)) !important;
      color: #00FFFF !important;
      font-weight: 600 !important;
    }

    /* TOC Items */
    #aside-content #card-toc .toc-item,
    #card-toc .toc-item {
      border-left: 1px solid rgba(0, 150, 200, 0.3) !important;
    }

    #aside-content #card-toc .toc-item.active,
    #card-toc .toc-item.active {
      border-left: 2px solid #00B4DC !important;
    }

    /* TOC Child items */
    #aside-content #card-toc .toc-child,
    #card-toc .toc-child {
      border-left: 1px solid rgba(0, 150, 200, 0.3) !important;
    }

    /* TOC Level 1 */
    #aside-content #card-toc .toc-level-1 > .toc-link,
    #card-toc .toc-level-1 > .toc-link {
      color: #A0D8F0 !important;
      font-weight: 600;
    }

    /* TOC Level 2 */
    #aside-content #card-toc .toc-level-2 > .toc-link,
    #card-toc .toc-level-2 > .toc-link {
      color: #8AB8C8 !important;
      padding-left: 16px;
    }

    /* TOC Level 3 */
    #aside-content #card-toc .toc-level-3 > .toc-link,
    #card-toc .toc-level-3 > .toc-link {
      color: #6A9CAC !important;
      padding-left: 28px;
    }

    /* TOC Percentage */
    #aside-content #card-toc .toc-percentage,
    #card-toc .toc-percentage {
      color: #6AB8C8 !important;
    }

    /* TOC Header / Title */
    #aside-content #card-toc .toc-title,
    #card-toc .toc-title,
    #aside-content #card-toc .card-widget-title,
    #card-toc .card-widget-title {
      color: #00D4FF !important;
      border-bottom: 1px solid rgba(0, 150, 200, 0.3) !important;
    }
  `;

  console.log('TOC card background applied');
}

// Try multiple times to ensure it runs
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setBackground();
    fixArticlePageBackground();
  });
} else {
  setBackground();
  fixArticlePageBackground();
}

window.addEventListener('load', () => {
  setBackground();
  fixArticlePageBackground();
  // Additional TOC fix on load
  if (document.getElementById('body-wrap')?.classList.contains('post')) {
    fixTOCCardBackground();
  }
});

setTimeout(setBackground, 100);
setTimeout(setBackground, 500);
setTimeout(fixArticlePageBackground, 100);
setTimeout(fixArticlePageBackground, 500);
setTimeout(fixTOCCardBackground, 200);
setTimeout(fixTOCCardBackground, 600);
