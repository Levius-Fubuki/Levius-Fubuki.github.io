# Levius Blog

静态博客与藏书架：<https://leviusspace.top/>。

## Codex 写作、维护与发布入口

先读 [AGENTS.md](AGENTS.md)，再按 [博客维护规格](BLOG_MAINTENANCE_SPEC.md) 操作。

规格覆盖来源核验、长文成熟度、栏目边界、正文来源、英文界面、书架与封面、生成与验证、发布授权及跨聊天交接。它不依赖本聊天历史；动态待发布状态以当前 Git 差异和可用维护 state 为准。

2026-10-08 起，用户持续授权后续博客更新在成熟度与检查通过后自动发布；完整流程及范围见规格，用户当次明确要求仅本地时遵循最新指令。

本机通常位于 `$HOME/Desktop/Idea/blogs_site`。在其他 Codex 聊天中指定这个项目，或提供 `BLOG_MAINTENANCE_SPEC.md` 的完整路径即可读取。该入口不会自动让仓库外的无关聊天加载本项目。

这是改造后的 Hexo 静态产物仓库，不能直接套用完整 Hexo 源站流程。普通维护不运行 `scripts/restyle-inner-pages.py`；其旧正文来源可能覆盖后来的更新。

## 本地预览

在仓库根目录运行：

```sh
python3 scripts/preview-server.py --port 4173
```

访问 `http://127.0.0.1:4173/?view=book-library#articles`。构建、依赖检查和发布前提详见规格；不要把启动预览当作发布成功。
