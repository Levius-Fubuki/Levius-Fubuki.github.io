"""Shared static book markup. Article bodies and taxonomy remain untouched."""
from pathlib import Path
from html import escape as esc
from urllib.parse import unquote, urlsplit
import json
from bs4 import BeautifulSoup as Soup

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'data/book-library.json').read_text())
BOOKS = DATA['books']
COLLECTIONS = {item['id']: item for item in DATA['collections']}
BY_PATH = {book['path']: book for book in BOOKS}


def find_book(href):
    return BY_PATH.get(unquote(urlsplit(href).path))


def cover(book, *, decorative=False):
    collection = COLLECTIONS[book['collection']]
    title = ''.join(f'<span>{esc(line)}</span>' for line in book['coverTitle'])
    alt = '' if decorative else f"《{book['title']}》暗黑中世纪插画封面"
    return f'''<span class="book-cover" data-collection="{book['collection']}">
      <img class="book-art" src="{esc(book.get('art', '/img/books/'+book['id']+'.webp'))}" alt="{esc(alt)}" width="640" height="960" loading="lazy" decoding="async" />
      <span class="book-foil" aria-hidden="true"></span>
      <span class="book-edition"><span>{collection['mark']} {esc(collection['title'])}</span><span>VOL. {book['volume']}</span></span>
      <span class="book-cover-type"><span class="book-cover-title">{title}</span><span class="book-cover-english">{esc(book['english'])}</span><span class="book-imprint">LEVIUS FUBUKI · COLLECTED NOTES</span></span>
    </span>'''


def shelf():
    items = []
    for book in BOOKS:
        collection = COLLECTIONS[book['collection']]
        items.append(f'''<article class="shelf-book" data-series="{book['collection']}" data-book="{book['id']}">
          <a class="book-link" href="{esc(book['path'])}" aria-label="阅读：{esc(book['title'])}">{cover(book, decorative=True)}</a>
          <div class="book-caption"><p class="book-series">{esc(collection['title'])} / VOL. {book['volume']}</p><h3><a href="{esc(book['path'])}">{esc(book['title'])}</a></h3></div>
        </article>''')
    spines = []
    for number, collection in enumerate(COLLECTIONS.values(), 1):
        members = [book for book in BOOKS if book['collection'] == collection['id']]
        art = members[0].get('art', '/img/books/'+members[0]['id']+'.webp')
        spines.append(f'''<button class="collection-spine" type="button" data-open-collection="{collection['id']}" data-title="{esc(collection['title'])}" data-subtitle="{esc(collection['subtitle'])}" data-description="{esc(collection['description'])}" data-href="{esc(collection['href'])}" aria-haspopup="dialog" aria-expanded="false" aria-label="展开{esc(collection['title'])}合集，共 {len(members)} 篇文章">
          <span class="spine-object" data-collection="{collection['id']}" aria-hidden="true"><span class="collection-front"><img class="collection-art-image" src="{esc(art)}" alt="" width="640" height="960" decoding="async" /><span class="collection-cover-shade"></span><span class="collection-cover-heading"><small>{collection['mark']} COLLECTION {number:02d}</small><strong>{esc(collection['title'])}</strong><span>{esc(collection['subtitle'])}</span><em>LEVIUS · COLLECTED NOTES</em></span></span><span class="collection-back"></span><span class="collection-top"></span><span class="collection-page-edge"></span><span class="spine-face"><span class="spine-cap">COLLECTION {number:02d}</span><span class="spine-emblem">{collection['mark']}</span><span class="spine-title">{esc(collection['title'])}</span><span class="spine-subtitle">{esc(collection['subtitle'])}</span><span class="spine-count">{len(members):02d}<small>VOLUMES</small></span><span class="spine-imprint">LEVIUS</span></span></span>
          <span class="spine-caption">{esc(collection['subtitle'])}<span>{esc(collection['title'])}</span></span>
        </button>''')
    reader_data = json.dumps(DATA, ensure_ascii=False).replace('<', '\\u003c')
    return f'''<div class="book-library">
      <section class="collection-shelf" aria-label="合集藏书架"><div class="cabinet-heading"><p>选择一部合集，翻阅其中的篇章。</p><span>{len(COLLECTIONS):02d} COLLECTIONS / {len(BOOKS):02d} VOLUMES</span></div><div class="spine-rack">{''.join(spines)}</div><div class="cabinet-plinth" aria-hidden="true"></div><p class="cabinet-inscription">LEVIUS’S LIBRARY <span>·</span> 藏于书脊之间</p></section>
      <section class="collection-detail" id="collection-volumes" aria-labelledby="collection-volume-title" hidden>
      <div class="shelf-toolbar"><button class="collection-return" type="button">← 返回藏书架</button><span class="shelf-count" aria-live="polite"></span></div>
      <div class="collection-detail-heading"><div><p class="book-series" id="collection-volume-subtitle"></p><h3 id="collection-volume-title" tabindex="-1"></h3></div><a class="collection-index-link" href="/categories/">查看合集索引 ↗</a></div>
      <div class="book-rail" id="book-rail" role="region" aria-label="文章藏书，可左右滚动" tabindex="0">{''.join(items)}</div>
      <div class="shelf-bottom"><span class="collection-description-line"></span><div class="shelf-controls"><button type="button" data-shelf-step="-1" aria-controls="book-rail" aria-label="上一组书籍">←</button><button type="button" data-shelf-step="1" aria-controls="book-rail" aria-label="下一组书籍">→</button></div></div>
      </section><script type="application/json" id="collection-reader-data">{reader_data}</script><noscript><p><a href="/categories/">浏览全部合集与文章 ↗</a></p></noscript>
    </div>'''


def article_cover(book):
    collection = COLLECTIONS[book['collection']]
    branch = f" · {book['branch']}" if book.get('branch') else ''
    return f'''<section class="article-book" aria-label="本篇封面">
      {cover(book)}
      <div class="article-book-caption"><a href="{esc(collection['href'])}">{collection['mark']} {esc(collection['title'])}{esc(branch)} ↗</a><span>第 {book['volume']} 卷</span></div>
    </section>'''


def enhance_list(content):
    soup = Soup(content, 'html.parser')
    for index, row in enumerate(soup.select('.article-sort-item:not(.year)')):
        link = row.select_one('.article-sort-item-title')
        book = find_book(link.get('href', '')) if link else None
        if not book:
            continue
        row['class'] = [c for c in row.get('class', []) if c != 'no-article-cover'] + ['has-book-preview']
        collection = COLLECTIONS[book['collection']]
        preview_id = f"book-preview-{index}"
        row.append(Soup(f'''<button class="book-preview-toggle" type="button" aria-expanded="false" aria-controls="{preview_id}" aria-label="预览《{esc(book['title'])}》"><span>预览</span><span class="preview-sign" aria-hidden="true">＋</span></button>
          <div class="archive-book-preview" id="{preview_id}" inert><div class="archive-preview-clip"><div class="archive-preview-body">
            <a class="archive-cover-link" href="{esc(book['path'])}" aria-label="阅读：{esc(book['title'])}">{cover(book, decorative=True)}</a>
            <div class="archive-book-copy"><span class="book-series">{collection['mark']} {esc(collection['title'])} / VOL. {book['volume']}</span><p>{esc(book['summary'])}</p><a class="book-read-link" href="{esc(book['path'])}">翻开这本书 <span aria-hidden="true">↗</span></a></div>
          </div></div></div>''', 'html.parser'))
    return str(soup)


def collections_page():
    blocks = []
    for series_id, collection in COLLECTIONS.items():
        if not collection['href'].startswith('/categories/'):
            continue
        collection = COLLECTIONS[series_id]
        books = [book for book in BOOKS if book['collection'] == series_id]
        stack = ''.join(f'<span class="collection-volume">{cover(book, decorative=True)}</span>' for book in books[:3][::-1])
        branch = collection.get('branch')
        branch_count = sum(book.get('branch') == branch['title'] for book in books) if branch else 0
        sequence = collection.get('sequence', '基础概念 → 价值 → 策略 → Actor-Critic → 树搜索' if series_id == 'rl' else '')
        branch_html = f'''<a class="collection-branch" href="{esc(branch['href'])}"><span class="branch-symbol" aria-hidden="true">└</span><span><b>{esc(branch['title'])}</b><small>{esc(branch['description'])}</small></span><span class="branch-count">{branch_count:02d} 卷 ↗</span></a>''' if branch else f'<p class="collection-sequence">{esc(sequence)}</p>'
        blocks.append(f'''<section class="collection-set" data-collection="{series_id}"><a class="collection-art" href="{esc(collection['href'])}" aria-label="浏览{esc(collection['title'])}合集">{stack}</a><div class="collection-info"><p class="book-series">{collection['mark']} {esc(collection['english'])}</p><h2><a href="{esc(collection['href'])}">{esc(collection['title'])}<span>{len(books):02d} 卷</span></a></h2><p class="collection-subtitle">{esc(collection['subtitle'])}</p><p class="collection-description">{esc(collection['description'])}</p>{branch_html}<a class="collection-open" href="{esc(collection['href'])}">浏览合集 ↗</a></div></section>''')
    return '<div id="page" class="collection-library">' + ''.join(blocks) + '</div>'
