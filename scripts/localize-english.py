#!/usr/bin/env python3
"""Publish English site chrome and metadata while preserving original article prose.

Run after any static generator. URLs/anchors remain stable. Translation dictionaries
are versioned under data/; source Markdown and the source book catalog stay intact.
"""
from pathlib import Path
from html.parser import HTMLParser
from html import escape, unescape
import copy, hashlib, json, re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r'[\u3400-\u9fff]')
SOURCE = json.loads((ROOT/'data/book-library.json').read_text())
CONTENT = json.loads((ROOT/'data/english-content.json').read_text())
UI = json.loads((ROOT/'data/english-ui.json').read_text())
TOC_PATH = ROOT/'data/english-toc.json'
MAPPING = dict(UI)
ENGLISH = copy.deepcopy(SOURCE)
for kind in ('books', 'collections'):
    for source, target in zip(SOURCE[kind], ENGLISH[kind]):
        changes = CONTENT[kind][source['id']]
        for key, value in changes.items():
            if isinstance(value, str):
                MAPPING[source[key]] = value
                target[key] = value
            elif isinstance(value, list):
                for old, new in zip(source[key], value):
                    # Shared short cover fragments must not override taxonomy labels.
                    MAPPING.setdefault(old, new)
                target[key] = value
            elif isinstance(value, dict):
                for subkey, text in value.items():
                    MAPPING[source[key][subkey]] = text
                    target[key][subkey] = text
if TOC_PATH.exists():
    for source, target in json.loads(TOC_PATH.read_text()).items():
        if source in MAPPING and target != MAPPING[source]:
            MAPPING[target] = MAPPING[source]
        MAPPING.setdefault(source, target)
# Prefer full titles and sentences over shorter label substitutions.
PATTERN = re.compile('|'.join(re.escape(k) for k in sorted(MAPPING, key=len, reverse=True)))

def translate(value):
    result = PATTERN.sub(lambda m: MAPPING[m[0]], value)
    result = re.sub(r'第 ([IVX0-9]+) 卷', r'Volume \1', result)
    result = re.sub(r'(\d+) 篇(?:文章|Articles)', r'\1 articles', result)
    result = re.sub(r'(\d+) 篇', r'\1 articles', result)
    result = re.sub(r'(\d+) 卷', r'\1 volumes', result)
    result = re.sub(r'展开(.+?)合集，共 (\d+) articles', r'Open \1 collection, \2 articles', result)
    result = re.sub(r'浏览(.+?)合集', r'Browse \1 collection', result)
    result = re.sub(r'《(.+?)》暗黑中世纪插画封面', r'Medieval illustrated cover for \1', result)
    result = result.replace('阅读：', 'Read: ').replace('Preview《', 'Preview: ').replace('》', '')
    result=result.replace('articlesArticles','articles').replace('！','.')
    result=result.replace('. Please credit the source when reusing this work.', '. When reusing this work, please credit')
    return re.sub(r'\b(0?1) (articles|volumes)\b', lambda m:m[1]+' '+m[2][:-1], result)

for book in ENGLISH['books']:
    if 'tags' in book: book['tags'] = [translate(x) for x in book['tags']]
    if 'branch' in book: book['branch'] = translate(book['branch'])
BY_PATH = {b['path']: b for b in ENGLISH['books']}
BODY = re.compile(r'<article\b[^>]*\bid=["\']article-container["\'][^>]*>.*?</article>', re.S)
TEXT_ATTRS = {'title','alt','aria-label','placeholder','data-title','data-subtitle','data-description'}
ATTR = re.compile(r'([\w:-]+)(\s*=\s*)(["\'])(.*?)\3', re.S)

class Localizer(HTMLParser):
    def __init__(self, source, book=None):
        super().__init__(convert_charrefs=False)
        self.source=source; self.book=book; self.edits=[]; self.raw=None; self.json_data=False
        self.offsets=[0]
        for match in re.finditer('\n',source): self.offsets.append(match.end())
    def edit(self, old, new):
        if old==new:return
        line,column=self.getpos();start=self.offsets[line-1]+column
        self.edits.append((start,start+len(old),new))
    def handle_starttag(self, tag, attrs):
        raw=self.get_starttag_text(); values=dict(attrs)
        def attr(m):
            name,eq,quote,value=m.groups(); decoded=unescape(value)
            if name in TEXT_ATTRS:new=translate(decoded)
            elif name in {'src','href'} and tag in {'script','link'} and decoded.startswith(('/js/','/css/')):
                asset=urlsplit(decoded).path; local=ROOT/asset.lstrip('/')
                if not local.is_file():return m[0]
                new=asset+'?v='+hashlib.sha256(local.read_bytes()).hexdigest()[:12]
            elif name=='lang' and tag=='html':new='en'
            elif name=='content' and tag=='meta':
                key=values.get('name',values.get('property',''))
                if key in {'description','og:description','twitter:description'} and self.book:new=self.book['summary']
                elif key=='og:locale':new='en_US'
                elif key in {'description','og:description','twitter:description','og:title','twitter:title','article:tag','keywords'}:new=translate(decoded)
                else:return m[0]
            else:return m[0]
            return name+eq+quote+escape(new,quote=True)+quote
        self.edit(raw,ATTR.sub(attr,raw))
        if tag in {'script','style'}:
            self.raw=tag;self.json_data=values.get('id')=='collection-reader-data'
    handle_startendtag=handle_starttag
    def handle_endtag(self,tag):
        if tag==self.raw:self.raw=None;self.json_data=False
    def handle_data(self,text):
        if self.json_data:
            self.edit(text,json.dumps(ENGLISH,ensure_ascii=False).replace('<','\\u003c'))
        elif not self.raw:self.edit(text,escape(translate(unescape(text)),quote=False))
    def result(self):
        self.feed(self.source)
        result=self.source
        for start,end,new in reversed(self.edits):result=result[:start]+new+result[end:]
        return result

def english_body(body, book):
    if not book:return body
    # Only the article's top-level title changes; prose, section headings and IDs remain exact.
    return re.sub(r'(<h1\b[^>]*>).*?(</h1>)',lambda m:m[1]+escape(book['title'])+m[2],body,count=1,flags=re.S)

def localize_page(path):
    source=path.read_text();book=BY_PATH.get('/'+str(path.relative_to(ROOT)).removesuffix('index.html'))
    protected=[]
    def hold(match):
        token=f'<!--ENGLISH_BODY_{len(protected)}-->'
        protected.append((token,english_body(match[0],book)))
        return token
    masked=BODY.sub(hold,source)
    output=Localizer(masked,book).result()
    # Cover line breaks are designed per book, independent of taxonomy labels.
    for source_book, english_book in zip(SOURCE['books'], ENGLISH['books']):
        old='<span class="book-cover-title">'+''.join('<span>'+escape(translate(line),quote=False)+'</span>' for line in source_book['coverTitle'])+'</span>'
        new='<span class="book-cover-title">'+''.join('<span>'+escape(line,quote=False)+'</span>' for line in english_book['coverTitle'])+'</span>'
        output=output.replace(old,new)
    if '/css/english-ui.css' not in output:
        revision=hashlib.sha256((ROOT/'css/english-ui.css').read_bytes()).hexdigest()[:12]
        output=output.replace('</head>',f'<link rel="stylesheet" href="/css/english-ui.css?v={revision}" />\n</head>')
    for token,body in protected:output=output.replace(token,body)
    if output!=source:path.write_text(output)
    return output!=source

def localize_feed(path):
    source=path.read_text();bodies=[]
    def protect(m):
        token=f'__PRESERVED_FEED_BODY_{len(bodies)}__';bodies.append((token,m[0]));return token
    masked=re.sub(r'<content\b[^>]*>.*?</content>',protect,source,flags=re.S)
    def summary(match):
        entry=match[0]
        link=re.search(r'<link\b[^>]*href=["\']([^"\']+)',entry)
        book=BY_PATH.get(unquote(urlsplit(unescape(link[1])).path)) if link else None
        if book:
            entry=re.sub(r'<summary\b[^>]*>.*?</summary>',lambda _: '<summary type="text">'+escape(book['summary'])+'</summary>',entry,flags=re.S)
        return entry
    masked=re.sub(r'<entry\b[^>]*>.*?</entry>',summary,masked,flags=re.S)
    # Translate only presentation fields; never identifiers, links, or content.
    masked=re.sub(r'<(title|subtitle|summary|category|tag)(\b[^>]*)>(.*?)</\1>',lambda m:'<'+m[1]+m[2]+'>'+escape(translate(unescape(m[3])),quote=False)+'</'+m[1]+'>',masked,flags=re.S)
    masked=re.sub(r'(<(?:category|tag)\b[^>]*\bterm=")([^"]*)(")',lambda m:m[1]+escape(translate(unescape(m[2])),quote=True)+m[3],masked)
    for token,body in bodies:masked=masked.replace(token,body)
    if masked!=source:path.write_text(masked)


def main():
    paths=public_pages()
    for name in ('search.xml','atom.xml'):localize_feed(ROOT/name)
    index=ROOT/'js/search-index.json'
    if index.exists():
        items=json.loads(index.read_text())
        for item in items:item['title']=translate(item['title'])
        index.write_text(json.dumps(items,ensure_ascii=False,separators=(',',':')))
        home_script=ROOT/'js/home.js'
        source=home_script.read_text()
        revision=hashlib.sha256(index.read_bytes()).hexdigest()[:12]
        home_script.write_text(re.sub(r'/js/search-index\.json(?:\?v=[a-f0-9]+)?','/js/search-index.json?v='+revision,source))
    changed=sum(localize_page(path) for path in paths)
    (ROOT/'data/book-library.en.json').write_text(json.dumps(ENGLISH,ensure_ascii=False,indent=2)+'\n')
    print(f'English localization: {len(paths)} pages checked, {changed} updated; article prose preserved.')

def public_pages():
    return sorted(p for p in ROOT.rglob('*.html') if p.relative_to(ROOT).parts[0] not in {'docs','card','card-studio','content','node_modules'} and not any(part.startswith('.') for part in p.relative_to(ROOT).parts))

if __name__=='__main__':main()
