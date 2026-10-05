"""Render registered Markdown posts and reconcile their static library indexes.

Run with Node's `marked` package available through NODE_PATH, plus beautifulsoup4.
Existing article bodies are read from the current checkout and never regenerated
from an old Git revision. Back up the checkout before the first publication run.
"""
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from html import escape as esc
from collections import defaultdict
import json,re,subprocess
from bs4 import BeautifulSoup as Soup
from book_library import DATA,BOOKS,COLLECTIONS,article_cover,enhance_list,collections_page

ROOT=Path(__file__).resolve().parents[1]
SITE='https://leviusspace.top'
AUTHORED=[b for b in BOOKS if b.get('source')]
ORDER=sorted(BOOKS,key=lambda b:b['date'],reverse=True)
changed=[]
modified_stamps={}

def read(path):return Soup((ROOT/path).read_text(),'html.parser')
def write(path,text):
    p=ROOT/path
    if p.exists() and p.read_text()==text:return
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);changed.append(str(path))
def replace_node(node,html):node.replace_with(Soup(html,'html.parser'))
def url(path):return SITE+quote(path,safe='/')
def metadata(s,title,path,description,art=None,date=None):
    s.title.string=title+" | L_F's Blog"
    s.select_one('link[rel="canonical"]')['href']=url(path)
    values={'description':description,'og:description':description,'twitter:description':description,'og:title':title,'twitter:title':title,'og:url':url(path)}
    if art:values.update({'og:image':SITE+art,'twitter:image':SITE+art})
    if date:values.update({'article:published_time':date,'article:modified_time':date})
    for m in s.select('meta'):
        k=m.get('name',m.get('property',''))
        if k in values:m['content']=values[k]
    for m in list(s.select('meta[property="article:tag"]')):m.decompose()

def new_toc(article):
    links=[];used={n['id'] for n in article.select('[id]') if n.name not in {'h2','h3'}}
    for i,h in enumerate(article.select('h2,h3'),1):
        ident=h.get('id') or re.sub(r'[^\w\u4e00-\u9fff-]+','-',h.get_text().strip()).strip('-').lower() or f'section-{i}'
        base=ident;suffix=2
        while ident in used:ident=f'{base}-{suffix}';suffix+=1
        h['id']=ident;used.add(ident)
        links.append(f'<li class="toc-item toc-level-{h.name[1:]}"><a class="toc-link" href="#{esc(h["id"])}"><span class="toc-text">{esc(h.get_text())}</span></a></li>')
    return '<div class="toc-content"><ol class="toc">'+''.join(links)+'</ol></div>'

# Snapshot the current article body for legacy preservation checks.
legacy_bodies={b['id']:str(read(b['path'].lstrip('/')+'index.html').select_one('#article-container')) for b in BOOKS if not b.get('source')}
template=(ROOT/'2026/04/07/0-大模型学习路线图/index.html').read_text()
for b in AUTHORED:
    source=ROOT/b['source'];assert source.is_file()
    if b.get('math'):
        rendered=subprocess.check_output(['node',str(ROOT/'scripts/render-markdown.cjs'),str(source),'--math'],text=True)
    else:
        rendered=subprocess.check_output(['node','-e',"const fs=require('fs'),{marked}=require('marked');process.stdout.write(marked(fs.readFileSync(process.argv[1],'utf8')))",str(source)],text=True)
    fragment=Soup(rendered,'html.parser');assert fragment.h1.get_text()==b['title']
    fragment.h1['id']=b['id'];toc=new_toc(fragment)
    s=Soup(template,'html.parser');stamp=b['date']+'T00:00:00+08:00'
    existing_path=ROOT/b['path'].lstrip('/')/'index.html'
    existing_modified=None
    if existing_path.is_file():
        existing_modified=Soup(existing_path.read_text(),'html.parser').select_one('meta[property="article:modified_time"]')
    modified=b.get('updated') or (existing_modified.get('content') if existing_modified else stamp)
    if len(modified)==10:modified+='T00:00:00+08:00'
    modified_stamps[b['id']]=modified
    metadata(s,b['title'],b['path'],b['summary'],b['art'],stamp)
    s.select_one('meta[property="article:modified_time"]')['content']=modified
    s.select_one('.inner-title').string=b['title']
    a=s.select_one('#article-container');a.clear();a.append(fragment)
    for node in s.select('#post-meta time'):
        value=modified if 'post-meta-date-updated' in node.get('class',[]) else stamp
        node['datetime']=value;node['title']=value[:10];node.string=value[:10]
    cat=s.select_one('#post-meta a.post-meta-categories');cat.string=COLLECTIONS[b['collection']]['title'];cat['href']=COLLECTIONS[b['collection']]['href']
    replace_node(s.select_one('.article-book'),article_cover(b));replace_node(s.select_one('.toc-content'),toc)
    copyright_link=s.select_one('.post-copyright__type a');copyright_link['href']=url(b['path']);copyright_link.string=url(b['path'])
    tags=s.select_one('.post-meta__tag-list');tags.clear()
    for tag in b['tags']:
        tags.append(Soup(f'<a class="post-meta__tags" href="/tags/{quote(tag)}/">{esc(tag)}</a>','html.parser'))
        s.head.append(s.new_tag('meta',attrs={'property':'article:tag','content':tag}))
    write(b['path'].lstrip('/')+'index.html',str(s))

# Taxonomy comes from current article metadata for legacy posts.
tags=defaultdict(list)
for b in BOOKS:
    page=read(b['path'].lstrip('/')+'index.html')
    btags=b.get('tags') or [a.get_text() for a in page.select('.post-meta__tags')]
    for tag in btags:tags[tag].append(b)

def listing(members,label,kind):
    rows=[];year=None
    for b in sorted(members,key=lambda x:x['date'],reverse=True):
        if b['date'][:4]!=year:
            year=b['date'][:4];rows.append(f'<div class="article-sort-item year">{year}</div>')
        rows.append(f'<div class="article-sort-item"><div class="article-sort-item-info"><div class="article-sort-item-time"><time class="post-meta-date-created" datetime="{b["date"]}">{b["date"]}</time></div><a class="article-sort-item-title" href="{esc(b["path"])}" title="{esc(b["title"])}">{esc(b["title"])}</a></div></div>')
    return enhance_list(f'<div id="{kind}"><div class="article-sort-title">{esc(label)} · {len(members)} 篇</div><div class="article-sort">'+''.join(rows)+'</div></div>')

def index_page(path,title,description,content,kind):
    seed={'archives':'archives/index.html','category':'categories/强化学习/index.html','tag':'tags/Actor-Critic/index.html'}[kind]
    s=read(seed);metadata(s,title,'/'+str(path).removesuffix('index.html'),description)
    s.select_one('.inner-title').string=title;s.select_one('.inner-description').string=description
    crumbs=s.select_one('.breadcrumbs');crumbs.clear();crumbs.append(Soup('<a href="/#profile">主页</a><span>/</span>'+esc(title),'html.parser'))
    target=s.select_one('.inner-content');target.clear();target.append(Soup(content,'html.parser'))
    write(path,str(s))

archive_paths={p.relative_to(ROOT) for p in (ROOT/'archives').rglob('index.html')}
for b in BOOKS:
    archive_paths.update({Path('archives')/b['date'][:4]/'index.html',Path('archives')/b['date'][:4]/b['date'][5:7]/'index.html'})
for path in sorted(archive_paths):
    parts=path.parts[1:-1];prefix='-'.join(parts);members=[b for b in ORDER if b['date'].startswith(prefix)]
    title='文章归档'+(' · '+prefix if prefix else '')
    index_page(path,title,'按时间翻阅项目实践与学习笔记。',listing(members,title,'archive'),'archives')
for col in COLLECTIONS.values():
    if col['href'].startswith('/categories/'):
        members=[b for b in BOOKS if b['collection']==col['id']]
        index_page(Path(col['href'].lstrip('/'))/'index.html',col['title'],col['description'],listing(members,col['title'],'category'),'category')
        if col.get('branch'):
            branch=col['branch'];members=[b for b in members if b.get('branch')==branch['title']]
            index_page(Path(branch['href'].lstrip('/'))/'index.html',branch['title'],branch['description'],listing(members,branch['title'],'category'),'category')
for tag,members in tags.items():
    index_page(Path('tags')/tag/'index.html',tag,'围绕同一问题的文章与实践。',listing(members,tag,'tag'),'tag')
s=read('categories/index.html');target=s.select_one('.inner-content');target.clear();target.append(Soup(collections_page(),'html.parser'));write('categories/index.html',str(s))
s=read('tags/index.html');target=s.select_one('.tag-cloud-list');target.clear()
for tag in tags:target.append(Soup(f'<a href="/tags/{quote(tag)}/">{esc(tag)}</a>','html.parser'))
write('tags/index.html',str(s))

# Neighbor links cover the full chronological sequence, including its old newest post.
for i,b in enumerate(ORDER):
    path=b['path'].lstrip('/')+'index.html';s=read(path);parts=[]
    for j,label in [(i-1,'较新一篇'),(i+1,'较早一篇')]:
        if 0<=j<len(ORDER):
            n=ORDER[j];parts.append(f'<a class="pagination-related" href="{esc(n["path"])}" title="{esc(n["title"])}"><div class="info"><div class="info-1"><div class="info-item-1">{label}</div><div class="info-item-2">{esc(n["title"])}</div></div></div></a>')
    nav=s.select_one('#pagination');markup='<nav class="pagination-post" id="pagination">'+''.join(parts)+'</nav>'
    if nav:replace_node(nav,markup)
    else:s.select_one('#post').append(Soup(markup,'html.parser'))
    write(path,str(s))
    if b['id'] in legacy_bodies:assert str(s.select_one('#article-container'))==legacy_bodies[b['id']]

subprocess.run(['python3',str(ROOT/'scripts/build-book-library.py')],cwd=ROOT,check=True)
# Dynamic sidebar counts; collection count differs from category count (a branch is a category).
category_count=sum(c['href'].startswith('/categories/') for c in COLLECTIONS.values())+sum(bool(c.get('branch')) for c in COLLECTIONS.values())
for path in ROOT.rglob('*.html'):
    if path.relative_to(ROOT).parts[0] in {'docs','card','card-studio','content','.git'}:continue
    text=path.read_text();s=Soup(text,'html.parser');dirty=False
    for a in s.select('.library-nav a'):
        count={'/archives/':len(BOOKS),'/categories/':category_count,'/tags/':len(tags)}.get(a.get('href'))
        if count is not None and a.span and a.span.get_text()!=f'{count:02d}':a.span.string=f'{count:02d}';dirty=True
    if dirty:write(path.relative_to(ROOT),str(s))

# Replace only authored entries; current legacy feed entries remain intact.
newpaths={b['path'] for b in AUTHORED}
for fn in ['search.xml','atom.xml']:
    text=(ROOT/fn).read_text()
    def keep(m):
        found=re.search(r'<(?:id|url)>(.*?)</(?:id|url)>',m[0]);path=unquote(urlsplit(found[1]).path) if found else ''
        return '' if path in newpaths else m[0]
    text=re.sub(r'<entry>.*?</entry>',keep,text,flags=re.S);entries=[]
    for b in AUTHORED:
        body=read(b['path'].lstrip('/')+'index.html').select_one('#article-container').decode_contents();assert ']]>' not in body
        title=esc(b['title']);path=quote(b['path'],safe='/');stamp=b['date']+'T00:00:00+08:00'
        content='<content type="html"><![CDATA['+body+']]></content>'
        if fn=='search.xml':
            cats='<categories><category>'+esc(COLLECTIONS[b['collection']]['title'])+'</category></categories>';terms='<tags>'+''.join('<tag>'+esc(t)+'</tag>' for t in b['tags'])+'</tags>'
            entries.append(f'<entry><title>{title}</title><link href="{path}"/><url>{path}</url>{content}{cats}{terms}</entry>')
        else:
            entries.append(f'<entry><title>{title}</title><link href="{SITE+path}"/><id>{SITE+path}</id><published>{stamp}</published><updated>{modified_stamps[b["id"]]}</updated>{content}<summary type="text">{esc(b["summary"])}</summary></entry>')
    pos=text.index('<entry>');text=text[:pos].rstrip()+'\n'+'\n'.join(entries)+'\n'+text[pos:]
    if fn=='atom.xml':
        latest=max([b['date']+'T00:00:00+08:00' for b in BOOKS]+list(modified_stamps.values()))
        text=re.sub(r'<updated>.*?</updated>','<updated>'+latest+'</updated>',text,count=1)
    write(fn,text)
subprocess.run(['python3',str(ROOT/'scripts/build-search-index.py')],cwd=ROOT,check=True)
# Add any newly public pages to both sitemaps without rewriting old URLs.
newurls=set()
for b in AUTHORED:newurls.add(url(b['path']))
for p in [*[ROOT/p for p in archive_paths],*[ROOT/c['href'].lstrip('/')/'index.html' for c in COLLECTIONS.values() if c['href'].startswith('/categories/')],*[ROOT/'tags'/tag/'index.html' for tag in tags]]:
    newurls.add(url('/'+str(p.relative_to(ROOT)).removesuffix('index.html')))
text=(ROOT/'sitemap.xml').read_text();existing={unquote(u).removesuffix('index.html') for u in re.findall(r'<loc>(.*?)</loc>',text)}
added=[u for u in sorted(newurls) if unquote(u).removesuffix('index.html') not in existing]
if added:
    latest=max(b['date'] for b in BOOKS)
    text=text.replace('</urlset>',''.join(f'<url><loc>{esc(u)}</loc><lastmod>{latest}</lastmod><changefreq>monthly</changefreq><priority>0.6</priority></url>' for u in added)+'\n</urlset>')
write('sitemap.xml',text)
text=(ROOT/'sitemap.txt').read_text();existing={unquote(u).removesuffix('index.html') for u in text.splitlines()}
write('sitemap.txt',text.rstrip()+'\n'+''.join(u+'\n' for u in sorted(newurls) if unquote(u).removesuffix('index.html') not in existing))
print(json.dumps({'authored_posts':len(AUTHORED),'books':len(BOOKS),'collections':len(COLLECTIONS),'tags':len(tags),'written_files':len(changed)},ensure_ascii=False))

# Keep regenerated public UI and metadata in English.
import subprocess as _localize_subprocess
_localize_subprocess.run(["python3", str(ROOT / "scripts/localize-english.py")], check=True)
