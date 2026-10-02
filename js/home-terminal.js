(() => {
  "use strict";
  const terminal = document.querySelector("#terminal");
  const input = document.querySelector("#terminal-input");
  const form = document.querySelector("#terminal-form");
  const output = document.querySelector("#terminal-output");
  if (!terminal || !input || !window.LeviusBlog) return;
  const announcement = document.querySelector("#terminal-announcement");
  const keyboard = document.querySelector("#terminal-keyboard");
  const keyboardToggle = document.querySelector(".terminal-keyboard-toggle");
  const indexState = document.querySelector("#terminal-index-state");
  const commands = ["help", "ls", "cd", "pwd", "search", "cat", "open", "run", "about", "history", "clear", "motion"];
  const directories = ["/", "/articles", "/reinforcement", "/deep-learning", "/archives", "/about"];
  const routes = { archives: "/archives/", gallery: "/Gallery/", github: "https://github.com/Levius-Fubuki" };
  let cwd = "/", history = [], historyAt = 0, draft = "", generation = 0;
  let busy = false, composing = false, records;
  form.hidden = false;
  document.querySelector(".terminal-tools").hidden = false;
  terminal.querySelectorAll("[data-terminal-command]").forEach(button => { button.disabled = false; });

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function write(target, text, className) { target.append(element("p", text, className)); }
  function link(target, text, href) {
    const node = element("a", text);
    node.href = href;
    if (new URL(href, location.href).origin !== location.origin) { node.target = "_blank"; node.rel = "noopener noreferrer"; }
    target.append(node);
    return node;
  }
  function scrollOutput() { output.scrollTop = output.scrollHeight; }
  function setBusy(value) {
    busy = value;
    form.dataset.busy = String(value);
    input.readOnly = value;
    form.querySelector("button").disabled = value;
  }
  function updatePath() {
    document.querySelector("#terminal-path").textContent = cwd;
    document.querySelector(".terminal-input-path").textContent = cwd === "/" ? "root" : cwd.slice(1);
    terminal.querySelector('[data-terminal-command="cd /articles"]').setAttribute("aria-current", String(cwd === "/articles"));
  }
  async function loadArticles() {
    if (records) return records;
    indexState.textContent = "Loading…";
    try {
      const articles = await window.LeviusBlog.loadIndex();
      records = articles.map((item, index) => ({ ...item, id: index + 1 }));
      indexState.textContent = `${records.length} ARTICLES / READY`;
      document.querySelector(".terminal-count").textContent = String(records.length).padStart(2, "0");
      return records;
    } catch {
      indexState.textContent = "Load failed";
      throw new Error("The article index is unavailable. Run the command again or browse the archives.");
    }
  }
  function resolvePath(value) {
    const path = value.startsWith("/") ? [] : cwd.split("/").filter(Boolean);
    for (const part of value.split("/")) {
      if (!part || part === ".") continue;
      if (part === "..") path.pop(); else path.push(part);
    }
    return `/${path.join("/")}`;
  }
  function parse(value) {
    const tokens = [];
    let word = "", quote = "", started = false;
    for (const char of value) {
      if (quote) {
        if (char === quote) quote = ""; else word += char;
        started = true;
      } else if (char === '"' || char === "'") { quote = char; started = true; }
      else if (/\s/.test(char)) { if (started) tokens.push(word); word = ""; started = false; }
      else { word += char; started = true; }
    }
    if (quote) throw new Error("Unclosed quote. Close it before running the command.");
    if (started) tokens.push(word);
    return tokens;
  }
  function articleResults(target, articles, terms = []) {
    for (const article of articles) {
      const row = element("div", undefined, "terminal-result");
      row.append(element("span", String(article.id).padStart(2, "0"), "terminal-result-number"));
      const content = element("div");
      link(content, article.title, article.url);
      if (terms.length) {
        const position = article.text.toLocaleLowerCase().indexOf(terms[0]);
        const start = Math.max(0, position - 30);
        content.append(element("small", `${start ? "…" : ""}${article.text.slice(start, start + 125)}${article.text.length > start + 125 ? "…" : ""}`));
      }
      row.append(content); target.append(row);
    }
    write(target, "cat <id> to preview · open <id> to read", "terminal-dim");
  }
  function about(target) {
    write(target, "Levius / L_F's Blog");
    write(target, "I'm Levius, currently focused on AI Infra and the systems behind modern AI.");
    link(target, "GitHub / Levius-Fubuki ↗", routes.github);
  }
  function help(target) {
    const rows = [
      ["ls [articles]", "List the current directory or all articles"],
      ["cd articles | cd /", "Change to the article or root directory; supports .."],
      ["pwd", "Show the current path"],
      ["search <keywords>", "Search titles and article text using one or more keywords"],
      ["cat <id>", "Preview article text; IDs stay the same after searching"],
      ["open <id>", "Open an article in this tab"],
      ["run articles.exe", "Launch the article view in the terminal"],
      ["open archives | gallery | github", "Open archives, gallery, or GitHub"],
      ["about | cat /about.txt", "Show the blog introduction"],
      ["motion on | off", "Toggle ambient motion on the homepage"],
      ["history | clear", "Show session history or clear the output"],
      ["help", "Show this help"],
    ];
    rows.forEach(([command, description]) => write(target, `${command}\n  ${description}`));
    write(target, "↑ ↓ History · Tab Complete · Esc Clear input · Ctrl+L Clear output · Ctrl+C Cancel", "terminal-dim");
  }
  async function dispatch(command, args, target, ticket) {
    const argument = args.join(" ");
    const current = () => ticket === generation;
    switch (command) {
      case "clear": clear(); break;
      case "help": help(target); break;
      case "pwd": write(target, cwd); break;
      case "about": about(target); break;
      case "history": history.forEach((item, i) => write(target, `${String(i + 1).padStart(2, "0")}  ${item}`)); break;
      case "cd": {
        const next = resolvePath(argument || "/");
        if (!directories.includes(next)) throw new Error(`Directory not found: ${argument}. Available: ${directories.join(", ")}.`);
        cwd = next; updatePath(); write(target, `Current directory: ${cwd}`);
        break;
      }
      case "ls": {
        const path = resolvePath(argument || ".");
        if (path === "/") { write(target, "articles/  reinforcement/  deep-learning/  archives/  about/\nabout.txt"); break; }
        if (!directories.includes(path)) throw new Error(`Directory not found: ${argument}. Try ls /articles.`);
        if (path === "/about") { write(target, "about.txt  about.exe"); break; }
        const articles = await loadArticles();
        const selection = path === "/reinforcement" ? articles.filter(a => /Reinforcement Learning|Value-Based|Policy-Based|Actor-Critic|Monte Carlo|强化学习|基于价值|基于策略|蒙特卡洛/i.test(a.title)) : path === "/deep-learning" ? articles.filter(a => /Deep Learning|深度学习/i.test(a.title)) : articles;
        if (current()) { write(target, `${path.slice(1)}.exe · ${selection.length} articles`); articleResults(target, selection); }
        break;
      }
      case "search": {
        if (!argument.trim()) throw new Error("Usage: search <keywords>, for example search reinforcement learning.");
        const articles = await loadArticles();
        if (!current()) break;
        const terms = argument.toLocaleLowerCase().trim().split(/\s+/);
        const matches = articles.filter(item => terms.every(term => `${item.title} ${item.text}`.toLocaleLowerCase().includes(term)))
          .sort((a, b) => Number(b.title.toLocaleLowerCase().includes(argument.toLocaleLowerCase())) - Number(a.title.toLocaleLowerCase().includes(argument.toLocaleLowerCase())));
        write(target, matches.length ? `${matches.length} articles found · ${argument}` : `No results for “${argument}”. Try reinforcement learning or deep learning.`);
        if (matches.length) articleResults(target, matches, terms);
        break;
      }
      case "cat":
      case "open": {
        if (command === "cat" && resolvePath(argument) === "/about.txt") { about(target); break; }
        if (command === "open" && Object.hasOwn(routes, argument)) { await window.LeviusScene.navigate(routes[argument], argument); break; }
        if (!/^\d+$/.test(argument)) throw new Error(`Usage: ${command} <article id>. Run ls articles to see the IDs.`);
        const articles = await loadArticles();
        if (!current()) break;
        const article = articles.find(item => item.id === Number(argument));
        if (!article) throw new Error(`Article ${argument} not found. Run ls articles to see valid IDs.`);
        if (command === "open") { await window.LeviusScene.navigate(article.url, "ARTICLE"); break; }
        write(target, `${String(article.id).padStart(2, "0")} / ${article.title}`);
        write(target, "Article preview", "terminal-dim");
        write(target, article.text.slice(0, 1200) + (article.text.length > 1200 ? "…" : ""));
        link(target, "Read full article ↗", article.url);
        break;
      }
      case "motion": {
        if (!window.LeviusMotion) throw new Error("Ambient motion is unavailable.");
        if (argument && !["on", "off"].includes(argument)) throw new Error("Usage: motion on or motion off.");
        if (argument) window.LeviusMotion.setEnabled(argument === "on");
        const state = window.LeviusMotion.getState();
        write(target, state.reduced ? "Reduced motion is enabled in your system settings. Ambient motion is off." : `Ambient motion is ${state.enabled ? "on" : "off"}.`);
        break;
      }
      case "run": {
        const program = argument.replace(/\.exe$/i, "");
        if (!["articles", "reinforcement", "deep-learning", "archives", "about"].includes(program)) throw new Error("Usage: run articles.exe to launch the article view.");
        await window.LeviusScene.launch(program);
        break;
      }
      default: throw new Error(`Unknown command: ${command}. Type help to see available commands.`);
    }
  }
  function clear() {
    generation++;
    setBusy(false);
    output.replaceChildren();
    announcement.textContent = "Terminal output cleared.";
  }
  async function execute(raw) {
    const value = raw.trim().slice(0, 500);
    if (!value || busy) return;
    if (history.at(-1) !== value) history.push(value);
    if (history.length > 100) history.shift();
    historyAt = history.length; draft = ""; input.value = "";
    if (value === "clear") { clear(); return; }
    const ticket = ++generation;
    const record = element("div", undefined, "terminal-record");
    const echo = element("p", undefined, "terminal-command-echo");
    echo.append(element("span", `levius-sh:${cwd === "/" ? "root" : cwd.slice(1)}$ `), document.createTextNode(value));
    const response = element("div", undefined, "terminal-response");
    record.append(echo, response); output.append(record);
    while (output.children.length > 100) output.firstElementChild.remove();
    setBusy(true); scrollOutput();
    try {
      const [command, ...args] = parse(value);
      if (!command) throw new Error("Enter a command, such as help.");
      await dispatch(command.toLocaleLowerCase(), args, response, ticket);
    } catch (error) {
      if (ticket === generation) write(response, error.message, "terminal-error");
    } finally {
      if (ticket === generation) {
        setBusy(false); scrollOutput();
        announcement.textContent = `${value} completed. ${response.querySelectorAll('.terminal-result').length ? `${response.querySelectorAll('.terminal-result').length} articles found.` : response.textContent.slice(0, 100)}`;
      }
    }
  }
  function recall(direction) {
    if (!history.length) return;
    if (historyAt === history.length) draft = input.value;
    historyAt = Math.max(0, Math.min(history.length, historyAt + direction));
    input.value = historyAt === history.length ? draft : history[historyAt];
    input.setSelectionRange(input.value.length, input.value.length);
  }
  function complete() {
    const value = input.value;
    if (!value || input.selectionStart !== value.length || input.selectionEnd !== value.length) return false;
    const options = [...commands, "ls articles", "cd articles", "cd /", "cd ..", "run articles.exe", "cat /about.txt", "open archives", "open gallery", "open github", "motion on", "motion off"];
    const matches = options.filter(option => option.startsWith(value.toLocaleLowerCase()));
    // Prefer completing the command itself before suggesting its arguments.
    const words = matches.filter(option => !option.includes(" "));
    const candidates = words.length ? words : matches;
    if (candidates.length === 1) { input.value = `${candidates[0]} `; return true; }
    if (candidates.length > 1) announcement.textContent = `Available completions: ${candidates.join(", ")}`;
    return false;
  }
  function cancel() {
    generation++; setBusy(false); input.value = "";
    const last = output.querySelector(".terminal-record:last-child .terminal-response");
    if (last && !last.textContent) write(last, "^C Response cancelled.", "terminal-dim");
    announcement.textContent = "Input cancelled.";
  }
  form.addEventListener("submit", event => { event.preventDefault(); if (!composing) execute(input.value); });
  input.addEventListener("compositionstart", () => { composing = true; });
  input.addEventListener("compositionend", () => { composing = false; });
  input.addEventListener("keydown", event => {
    if (event.isComposing || composing || event.keyCode === 229) return;
    const control = event.ctrlKey || event.metaKey;
    if (control && event.key.toLowerCase() === "l") { event.preventDefault(); clear(); return; }
    if (control && event.key.toLowerCase() === "c" && input.selectionStart === input.selectionEnd) { event.preventDefault(); cancel(); return; }
    if (control && event.key.toLowerCase() === "k") { event.preventDefault(); input.value = input.value.slice(0, input.selectionStart); return; }
    if (busy) return;
    if (event.key === "ArrowUp" || event.key === "ArrowDown") { event.preventDefault(); recall(event.key === "ArrowUp" ? -1 : 1); }
    if (event.key === "Tab" && !event.shiftKey && complete()) event.preventDefault();
    if (event.key === "Escape") { event.preventDefault(); input.value = ""; }
    if (!keyboard.hidden) {
      const key = [...keyboard.querySelectorAll("button")].find(button => button.dataset.terminalKey === event.key.toLowerCase() || button.dataset.terminalKey === event.key);
      if (key) { key.classList.add("is-pressed"); setTimeout(() => key.classList.remove("is-pressed"), 130); }
    }
  });
  terminal.querySelectorAll("[data-terminal-command]").forEach(button => button.addEventListener("click", async () => {
    if (button.hasAttribute("data-launch")) return;
    if (busy) return;
    input.focus({ preventScroll: true });
    await execute(button.dataset.terminalCommand);
    if (button.dataset.terminalCommand === "cd /articles" && cwd === "/articles" && !busy) await execute("ls");
  }));
  document.querySelectorAll(".terminal-launch").forEach(button => button.addEventListener("click", () => {
    input.focus({ preventScroll: true });
  }));
  const rows = [
    ["Escape", ..."1234567890", "_", "Backspace"], ["Tab", ..."qwertyuiop", "()"],
    ["CapsLock", ..."asdfghjkl", ";", "Enter"], ["Shift", ..."zxcvbnm", ",", ".", "/", "ShiftRight"], [" "],
  ];
  let caps = false, shifted = false;
  const labels = { Escape: "ESC", Tab: "TAB", CapsLock: "CAPS", Shift: "SHIFT", ShiftRight: "SHIFT", " ": "SPACE", Backspace: "BACK", Enter: "ENTER" };
  for (const keys of rows) {
    const row = element("div", undefined, "terminal-key-row");
    for (const key of keys) {
      const button = element("button", labels[key] || key.toUpperCase());
      button.type = "button"; button.dataset.terminalKey = key;
      button.setAttribute("aria-label", key === " " ? "Space" : key === "Backspace" ? "Backspace" : labels[key] || key);
      // Preserve the native input's caret and selection when using the keyboard.
      button.addEventListener("pointerdown", event => { if (event.pointerType === "mouse") event.preventDefault(); });
      button.addEventListener("click", () => {
        input.focus({ preventScroll: true });
        if (busy) return;
        if (key === "Enter") { execute(input.value); return; }
        if (key === "Escape") { input.value = ""; return; }
        if (key === "Tab") { complete(); return; }
        if (key === "CapsLock") { caps = !caps; button.setAttribute("aria-pressed", String(caps)); return; }
        if (key === "Shift" || key === "ShiftRight") { shifted = !shifted; button.setAttribute("aria-pressed", String(shifted)); return; }
        const start = input.selectionStart, end = input.selectionEnd;
        if (key === "Backspace") input.setRangeText("", start === end ? Math.max(0, start - 1) : start, end, "end");
        else if (input.value.length - (end - start) + key.length <= input.maxLength) {
          input.setRangeText(caps !== shifted ? key.toUpperCase() : key, start, end, "end");
          shifted = false;
          keyboard.querySelectorAll('[data-terminal-key^="Shift"]').forEach(button => button.setAttribute("aria-pressed", "false"));
        }
      });
      row.append(button);
    }
    keyboard.append(row);
  }
  keyboardToggle.addEventListener("click", () => {
    keyboard.hidden = !keyboard.hidden;
    keyboardToggle.setAttribute("aria-expanded", String(!keyboard.hidden));
    keyboardToggle.textContent = keyboard.hidden ? "On-screen keyboard +" : "Hide keyboard −";
  });
  let launching = false;
  document.querySelectorAll("[data-launch]").forEach(button => button.addEventListener("click", async () => {
    if (launching || busy || window.LeviusScene.getState() !== "terminal") return;
    launching = true;
    try {
      for (const command of [`cd /${button.dataset.launch}`, "ls", `run ${button.dataset.launch}.exe`]) {
        input.value = "";
        if (!matchMedia("(prefers-reduced-motion: reduce)").matches && window.LeviusMotion.getState().enabled) {
          input.readOnly = true;
          for (const letter of command) {
            input.value += letter;
            const key = [...keyboard.querySelectorAll("button")].find(node => node.dataset.terminalKey === letter);
            if (key) { key.classList.add("is-pressed"); setTimeout(() => key.classList.remove("is-pressed"), 130); }
            await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 150));
          }
          input.readOnly = false;
        }
        await execute(command);
      }
    } finally { launching = false; input.readOnly = false; }
  }));
  window.LeviusTerminal = Object.freeze({ execute });
  // Loading starts near the workbench; the above-the-fold card keeps priority.
  const visibility = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    visibility.disconnect();
    loadArticles().catch(() => {});
  }, { rootMargin: "160px" });
  visibility.observe(terminal);
})();
