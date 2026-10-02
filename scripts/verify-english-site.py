#!/usr/bin/env python3
"""Audit English UI coverage, stable URLs and unchanged article prose."""
from pathlib import Path
from bs4 import BeautifulSoup as Soup
from urllib.parse import unquote,urlsplit
import json,re,subprocess,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1]
BASE='7ba12ab'
CJK=re.compile(r'[\u3400-\u9fff]')
content=json.loads((ROOT/'data/english-content.json').read_text())
catalog=json.loads((ROOT/'data/book-library.en.json').read_text())
books={b['path']:b for b in catalog['books']}
paths=[p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p.endswith('.html') and p.split('/')[0] not in {'docs','card','card-studio'}]
checked=0
for name in paths:
 source=(ROOT/name).read_text();s=Soup(source,'html.parser');assert s.html.get('lang')=='en',name
 body=s.select_one('#article-container')
 if body:
  old=Soup(subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT,text=True),'html.parser').select_one('#article-container')
  title=body.find('h1');original_title=old.find('h1')
  if title:title.clear()
  if original_title:original_title.clear()
  assert str(body)==str(old),f'Article content changed: {name}'
  checked+=1
  body.decompose()
 for e in s.select('script,style'):e.decompose()
 for text in s.stripped_strings:assert not CJK.search(text),(name,text)
 for e in s.find_all():
  for key in ['title','alt','aria-label','placeholder','data-title','data-subtitle','data-description']:
   assert not CJK.search(e.get(key,'')),(name,key,e.get(key))
  if e.name=='meta' and e.get('name',e.get('property')) in {'description','og:description','twitter:description','og:title','twitter:title','article:tag'}:
   assert not CJK.search(e.get('content','')),(name,e)
 for a in s.select('a[href]'):
  link=urlsplit(a['href'])
  if link.netloc:continue
  path=unquote(link.path)
  if not path.startswith('/') or path.startswith('//'):continue
  target=ROOT/path.lstrip('/')
  assert target.exists(),(name,'Missing target',path)
for book in catalog['books']:
 s=Soup((ROOT/book['path'].lstrip('/')/'index.html').read_text(),'html.parser')
 assert s.select_one('.inner-title').get_text()==content['books'][book['id']]['title'],book['id']
 assert s.select_one('meta[name="description"]')['content']==book['summary'],book['id']
items=json.loads((ROOT/'js/search-index.json').read_text())
assert len(items)==len(catalog['books'])==23
assert all(not CJK.search(item['title']) for item in items)
assert any('强化学习' in item['text'] for item in items),'Original Chinese article text must remain searchable'
# Localizing a second time must not change any output.
files=[ROOT/p for p in paths]+[ROOT/'atom.xml',ROOT/'search.xml',ROOT/'js/search-index.json',ROOT/'data/book-library.en.json']
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
subprocess.run(['python3',str(ROOT/'scripts/localize-english.py')],check=True)
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p)] for p in files),'Localization is not idempotent'
print(f'PASS: {len(paths)} English pages, {checked} preserved article bodies, 23 English search titles, valid navigation targets, and idempotent localization.')
