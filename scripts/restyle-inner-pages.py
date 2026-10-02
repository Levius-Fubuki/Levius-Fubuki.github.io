"""Apply the homepage shell to tracked Hexo output without editing article content.
Reads original pages from the recorded Git revision, so reruns are deterministic.
Requires beautifulsoup4. Run from the repository root.
"""
from pathlib import Path
from bs4 import BeautifulSoup as Soup
import subprocess, json, html, hashlib
from book_library import find_book, article_cover, enhance_list, collections_page
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/design/inner-pages/manifest.json'
MANIFEST.parent.mkdir(parents=True,exist_ok=True)
base=json.loads(MANIFEST.read_text())['source_revision'] if MANIFEST.exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
paths=[p for p in subprocess.check_output(['git','ls-tree','-rz','--name-only',base],cwd=ROOT,text=True).split('\0') if p.endswith('/index.html') and p.split('/')[0] in ['2025','2026','archives','categories','tags','Gallery']]
home=Soup((ROOT/'index.html').read_text(),'html.parser')
header=home.select_one('.site-header')
header.select_one('.main-nav a[href="/"]')['href']='/#profile'
for a in header.select('.main-nav a'):
 a.attrs.pop('aria-current',None)
 if a.get('href')=='#articles':a['href']='/#articles'
dialog=str(home.select_one('#search-dialog'))
footer=str(home.select_one('.site-footer'))
libnav='''<nav class="library-nav" aria-label="文章索引导航"><a href="/archives/">全部文章 <span>08</span></a><a href="/categories/">分类 <span>03</span></a><a href="/tags/">标签 <span>14</span></a><a href="/Gallery/">画廊 <span>↗</span></a></nav>'''
def sidebar(toc=None):
 if toc:
  return f'<aside class="inner-sidebar"><details class="contents-panel" open><summary>文章目录 <span>CONTENTS</span></summary>{toc}</details><a class="back-terminal" href="/#terminal">↙ 返回站内终端</a></aside>'
 return f'''<aside class="inner-sidebar"><section class="index-panel"><h2>探索索引 <span>/ INDEX</span></h2>{libnav}<div class="index-note"><b>LEVIUS / FUBUKI</b><p lang="en">From papers to implementations, with questions along the way.</p><p lang="en">Main focus: AI Infra<br>Also exploring multimodal inference and AI agents</p><a href="https://github.com/Levius-Fubuki" target="_blank" rel="noopener noreferrer">GitHub ↗</a></div></section><a class="back-terminal" href="/#terminal">↙ 返回站内终端</a></aside>'''
def clean_shell(node):
 for n in node.select('script,.post-share,.cover,.info-2,i[class*="fa"]'):n.decompose()
 for n in node.select('.tag-cloud-list a'):n.attrs.pop('style',None)
 return str(node)
manifest={'source_revision':base,'pages':[]}
for path in paths:
 source=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=ROOT,text=True)
 s=Soup(source,'html.parser');post=s.select_one('#post');kind='post' if post else path.split('/')[0].lower()
 old_article=s.select_one('#article-container')
 body_hash=hashlib.sha256(str(old_article).encode()).hexdigest() if old_article else None
 title=s.select_one('.post-title' if post else '#site-title').get_text(strip=True)
 if path=='archives/index.html':title='文章归档'
 elif path=='categories/index.html':title='分类索引'
 elif path=='tags/index.html':title='标签索引'
 elif kind=='gallery':title='画廊'
 if kind=='archives':label='ARCHIVE / LEVIUS NOTES';intro='按时间翻阅笔记，从一个问题走向下一次实践。'
 elif kind=='categories':label='CATEGORIES / KNOWLEDGE MAP';intro='沿着主题，找到相关的学习记录。'
 elif kind=='tags':label='TAGS / CROSS REFERENCES';intro='用关键词串起概念、算法与实现。'
 elif kind=='gallery':label='GALLERY / VISUAL ARCHIVE';intro='收集画面，也留下一些想象。'
 else:label='ARTICLE / LEVIUS NOTES';intro=''
 head=s.head
 for n in list(head.select('script,link[rel="stylesheet"],link[rel="preconnect"]')):n.decompose()
 head.select_one('meta[name="theme-color"]')['content']='#101010'
 for n in head.select('meta[property="og:image"],meta[name="twitter:image"]'):n['content']='https://leviusspace.top/img/redesign/portrait-cutout.webp'
 for stylesheet in ['/css/home.css','/css/katex/katex.min.css','/css/inner-pages.css','/css/book-library.css']:
  n=s.new_tag('link',rel='stylesheet',href=stylesheet);head.append(n)
 for script in ['/js/home.js','/js/inner-pages.js','/js/book-library.js']:
  n=s.new_tag('script',src=script,defer='');head.append(n)
 h=Soup(str(header),'html.parser')
 active='/Gallery/' if kind=='gallery' else '/archives/'
 link=h.select_one(f'.main-nav a[href="{active}"]')
 if link:link['aria-current']='page'
 if post:
  meta=s.select_one('#post-meta');second=meta.select_one('.meta-secondline')
  if second:second.decompose()
  for icon in meta.select('i'):icon.decompose()
  tools='<div class="reading-tools"><button class="reading-width" type="button" aria-pressed="false">专注阅读</button><button class="copy-page" type="button">复制链接</button></div>'
  title_extra=str(meta)+tools
  content=clean_shell(post)
  toc=s.select_one('#card-toc .toc-content')
  aside=sidebar(str(toc)) if toc else sidebar()
  book=find_book('/'+path.removesuffix('index.html'))
  if book:
   aside=aside.replace('<aside class="inner-sidebar">','<aside class="inner-sidebar">'+article_cover(book),1)
   for n in head.select('meta[property="og:image"],meta[name="twitter:image"]'):n['content']='https://leviusspace.top' + book.get('art', '/img/books/'+book['id']+'.webp')
 else:
  title_extra=f'<p class="inner-description">{intro}</p>'
  primary=s.select_one('#content-inner').find(recursive=False)
  content=clean_shell(primary)
  content=collections_page() if path=='categories/index.html' else enhance_list(content)
  if kind=='gallery':content='''<section class="gallery-empty"><div class="empty-frame" aria-hidden="true"><span>［ ＋ ］</span></div><h2>画廊尚未发布作品</h2><p>这里将用于展示图像与视觉创作。</p><a class="button secondary" href="/#profile">返回个人主页 ↗</a></section>'''
  aside=sidebar()
 crumbs='<a href="/#profile">主页</a><span>/</span><a href="/archives/">文章档案</a>' if post else '<a href="/#profile">主页</a><span>/</span>'+html.escape(title)
 result=f'''<!doctype html><html lang="zh-CN">{head}<body class="inner-page page-{kind}"><a class="skip-link" href="#main-content">跳到正文</a><div class="site-shell">{h}<main id="main-content"><nav class="breadcrumbs" aria-label="当前位置">{crumbs}</nav><header class="inner-masthead"><div class="inner-window-label"><span>{label}</span><span aria-hidden="true">□ □ □</span></div><div class="inner-title-block"><p class="inner-eyebrow">L_F'S BLOG / PERSONAL ARCHIVE</p><h1 class="inner-title" data-decode>{html.escape(title)}</h1>{title_extra}</div></header><div class="inner-layout"><div class="inner-content">{content}</div>{aside}</div></main>{footer}</div>{dialog}<button class="back-top" type="button" aria-label="回到顶部" hidden>↑</button><p class="sr-only" id="inner-status" role="status"></p><div class="reading-progress" aria-hidden="true"></div></body></html>'''
 out=ROOT/path;out.write_text(result)
 rendered=Soup(result,'html.parser').select_one('#article-container')
 if post and old_article and hashlib.sha256(str(rendered).encode()).hexdigest()!=body_hash:raise RuntimeError(f'Article body changed: {path}')
 manifest['pages'].append({'path':path,'kind':kind,'title':title,'article_sha256':body_hash,'math_count':len(old_article.select('.katex')) if old_article else 0,'images':len(old_article.select('img')) if old_article else 0})
MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(f'Restyled {len(paths)} pages; article HTML verified unchanged.')
# A branded GitHub Pages fallback uses the same navigation and search.
missing=Soup((ROOT/'Gallery/index.html').read_text(),'html.parser')
missing.title.string="页面不存在 | L_F's Blog"
missing.body['class']=['inner-page','page-notfound']
missing.select_one('link[rel="canonical"]')['href']='https://leviusspace.top/404.html'
for n in missing.select('meta[property="og:url"]'):n['content']='https://leviusspace.top/404.html'
for n in missing.select('meta[property="og:title"]'):n['content']='页面不存在'
for n in missing.select('[aria-current]'):n.attrs.pop('aria-current',None)
missing.select_one('.inner-title').string='404 / 页面不存在'
missing.select_one('.inner-description').string='这条路径没有对应的页面。'
missing.select_one('.breadcrumbs').clear()
missing.select_one('.breadcrumbs').append(Soup('<a href="/">主页</a><span>/</span>404','html.parser'))
missing.select_one('.inner-window-label span').string='NOT FOUND / RETURN TO INDEX'
missing.select_one('.empty-frame span').string='404'
missing.select_one('.gallery-empty h2').string='换一条路径，继续探索'
missing.select_one('.gallery-empty p').string='返回首页，或从文章归档中查找内容。'
(ROOT/'404.html').write_text(str(missing))

# Keep regenerated public UI and metadata in English.
import subprocess as _localize_subprocess
_localize_subprocess.run(["python3", str(ROOT / "scripts/localize-english.py")], check=True)
