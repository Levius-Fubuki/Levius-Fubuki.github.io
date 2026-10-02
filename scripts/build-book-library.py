"""Update only the homepage's book shelf and title; restyle inner pages separately."""
from pathlib import Path
import re
from book_library import shelf

ROOT = Path(__file__).resolve().parents[1]
page = ROOT / 'index.html'
text = page.read_text()
text = re.sub(r'<h2 id="opening-title"[^>]*>.*?</h2>', '<h2 id="opening-title" aria-label="Levius_Fubuki\'s BLOGS"><span>Levius_Fubuki\'s</span> <span>BLOGS</span></h2>', text, flags=re.S)
start = text.index('<section\n          id="articles"')
depth = 0
for match in re.finditer(r'</?section\b[^>]*>', text[start:]):
    depth += -1 if match.group().startswith('</') else 1
    if depth == 0:
        end = start + match.start()
        break
section = text[start:end]
heading_end = section.index('</div>') + len('</div>')
text = text[:start] + section[:heading_end] + '\n' + shelf() + '\n' + text[end:]
if '/css/book-library.css' not in text:
    text = text.replace('</head>', '<link rel="stylesheet" href="/css/book-library.css" />\n<script src="/js/book-library.js" defer></script>\n</head>')
if '/js/collection-reader.js' not in text:
    text = text.replace('<script src="/js/book-library.js"', '<script src="/js/collection-reader.js" defer></script>\n<script src="/js/book-library.js"')
page.write_text(text)
print('Homepage: opening title, collection cabinet and article volumes updated.')

# Keep regenerated public UI and metadata in English.
import subprocess as _localize_subprocess
_localize_subprocess.run(["python3", str(ROOT / "scripts/localize-english.py")], check=True)
