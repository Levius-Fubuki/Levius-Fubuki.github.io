"""Serve local previews without stale HTML/CSS or unversioned page navigation.
Usage: python3 scripts/preview-server.py --port 4173
Preview-only URL rewriting never modifies the site's generated files.
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import argparse, io, re, time
ROOT=Path(__file__).resolve().parents[1]
REVISION=str(time.time_ns())
ATTRIBUTE=re.compile(r'\b(href|src)=("|\')(/[^"\']*)\2', re.I)
def versioned(match):
    attr,quote,value=match.groups()
    if value.startswith('//'):return match.group(0)
    url=urlsplit(value)
    if not (url.path.endswith('/') or url.path.endswith(('.html','.css','.js'))):return match.group(0)
    query=dict(parse_qsl(url.query,keep_blank_values=True));query['__preview']=REVISION
    value=urlunsplit(('', '',url.path,urlencode(query),url.fragment))
    return f'{attr}={quote}{value}{quote}'
class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
    def end_headers(self):
        self.send_header('Cache-Control','no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma','no-cache')
        self.send_header('Expires','0')
        self.send_header('X-Levius-Preview',REVISION)
        super().end_headers()
    def send_head(self):
        for name in ['If-Modified-Since','If-None-Match']:
            if name in self.headers:del self.headers[name]
        file=Path(self.translate_path(self.path))
        if file.is_dir() and urlsplit(self.path).path.endswith('/'):file=file/'index.html'
        if file.is_file() and file.suffix=='.html':
            document=ATTRIBUTE.sub(versioned,file.read_text())
            script=f'<script src="/js/preview-navigation.js?__preview={REVISION}" data-preview-revision="{REVISION}" defer></script>'
            document=document.replace('</head>',script+'</head>',1)
            data=document.encode('utf-8')
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers()
            return io.BytesIO(data)
        return super().send_head()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=4173);args=parser.parse_args()
    print(f'Levius preview: http://127.0.0.1:{args.port}/?__preview={REVISION}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),PreviewHandler).serve_forever()
