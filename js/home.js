(() => {
  "use strict";
  const menuButton = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".main-nav");
  const dialog = document.querySelector("#search-dialog");
  const input = document.querySelector("#search-input");
  const results = document.querySelector("#search-results");
  const status = document.querySelector("#search-status");
  const searchButtons = document.querySelectorAll(".search-open");
  let index;
  let pendingIndex;
  let lastFocus;
  let timer;

  document.documentElement.classList.add("js-ready");
  if (menuButton) menuButton.hidden = false;
  searchButtons.forEach((button) => {
    button.hidden = false;
  });

  function closeMenu() {
    if (!nav || !menuButton) return;
    nav.classList.remove("is-open");
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "展开导航");
  }
  menuButton?.addEventListener("click", () => {
    const open = !nav.classList.contains("is-open");
    nav.classList.toggle("is-open", open);
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "收起导航" : "展开导航");
  });
  nav
    ?.querySelectorAll("a")
    .forEach((link) => link.addEventListener("click", closeMenu));

  async function loadIndex() {
    if (index) return index;
    if (!pendingIndex) {
      pendingIndex = fetch("/js/search-index.json")
        .then((response) => {
          if (!response.ok) throw new Error("Search index unavailable");
          return response.json();
        })
        .then((items) => {
          index = items.filter((item) => {
            if (
              typeof item.title !== "string" ||
              typeof item.text !== "string" ||
              typeof item.url !== "string"
            )
              return false;
            const url = new URL(item.url, location.href);
            return (
              url.origin === location.origin && url.pathname.startsWith("/")
            );
          });
          return index;
        })
        .catch((error) => {
          pendingIndex = undefined;
          throw error;
        });
    }
    return pendingIndex;
  }

  async function search() {
    const query = input.value.trim().toLocaleLowerCase();
    results.replaceChildren();
    if (!query) {
      status.textContent = "输入关键词，查找学习笔记。";
      return;
    }
    status.textContent = "正在搜索…";
    try {
      const articles = await loadIndex();
      if (input.value.trim().toLocaleLowerCase() !== query || !dialog.open)
        return;
      const terms = query.split(/\s+/);
      const matches = articles
        .filter((item) => {
          const text = `${item.title} ${item.text}`.toLocaleLowerCase();
          return terms.every((term) => text.includes(term));
        })
        .sort(
          (a, b) =>
            Number(b.title.toLocaleLowerCase().includes(query)) -
            Number(a.title.toLocaleLowerCase().includes(query)),
        );
      status.textContent = matches.length
        ? `找到 ${matches.length} 篇文章`
        : "没有找到相关文章，试试「强化学习」或「深度学习」。";
      const fragment = document.createDocumentFragment();
      for (const item of matches) {
        const li = document.createElement("li");
        const link = document.createElement("a");
        link.href = item.url;
        link.className = "search-result";
        const title = document.createElement("strong");
        title.textContent = item.title;
        const excerpt = document.createElement("p");
        const foundAt = item.text.toLocaleLowerCase().indexOf(terms[0]);
        const start = Math.max(0, foundAt - 35);
        excerpt.textContent = `${start ? "…" : ""}${item.text.slice(start, start + 115)}${item.text.length > start + 115 ? "…" : ""}`;
        link.append(title, excerpt);
        li.append(link);
        fragment.append(li);
      }
      results.replaceChildren(fragment);
    } catch {
      if (input.value.trim().toLocaleLowerCase() === query)
        status.textContent =
          "搜索暂时无法加载，请重新输入重试，或前往归档浏览。";
    }
  }

  // Shared with the homepage terminal; one cached request for both interfaces.
  window.LeviusBlog = Object.freeze({ loadIndex });

  function openSearch() {
    if (dialog.open) return;
    lastFocus = document.activeElement;
    dialog.showModal();
    input.focus();
    search();
  }
  searchButtons.forEach((button) =>
    button.addEventListener("click", openSearch),
  );
  document
    .querySelector(".search-close")
    .addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    const rect = dialog.getBoundingClientRect();
    if (
      event.target === dialog &&
      (event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom)
    )
      dialog.close();
  });
  dialog.addEventListener("close", () => {
    clearTimeout(timer);
    if (lastFocus instanceof HTMLElement) lastFocus.focus();
  });
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(search, 120);
  });
  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.target.closest?.("#terminal")) return;
    if (event.key === "Escape" && dialog.open) {
      event.preventDefault();
      dialog.close();
      return;
    }
    const isEditing =
      event.target instanceof HTMLElement &&
      (event.target.matches("input, textarea, select") ||
        event.target.isContentEditable);
    if (
      (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) ||
      (event.key === "/" && !isEditing)
    ) {
      event.preventDefault();
      openSearch();
    }
    if (
      event.key === "Escape" &&
      !dialog.open &&
      nav?.classList.contains("is-open")
    ) {
      closeMenu();
      menuButton.focus();
    }
  });
})();

// Stop the card's render loop while the hero is off screen or behind search.
(() => {
  const frame = document.querySelector('.identity-card');
  if (!frame) return;
  let inView = true;
  const sync = () => frame.contentWindow?.postMessage({
    type: 'levius-card-visibility',
    visible: inView && !document.hidden && !document.querySelector('#search-dialog')?.open,
  }, location.origin);
  new IntersectionObserver(([entry]) => { inView = entry.isIntersecting; sync(); }).observe(frame);
  frame.addEventListener('load', sync);
  document.addEventListener('visibilitychange', sync);
  const searchDialog = document.querySelector('#search-dialog');
  if (searchDialog) new MutationObserver(sync).observe(searchDialog, { attributes:true, attributeFilter:['open'] });
})();
