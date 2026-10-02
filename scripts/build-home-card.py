"""Publish only the identity card runtime and referenced assets into /card/."""
from pathlib import Path
import json, shutil, subprocess, hashlib, re
root=Path(__file__).resolve().parents[1]
source=root/'card-studio/levius-id/web';dest=root/'card'
subprocess.run(['npx','--yes','esbuild@0.25.0',str(source/'app.js'),'--bundle','--format=esm','--minify','--target=es2020','--outfile='+str(source/'app.bundle.js')],check=True)
dest.mkdir(exist_ok=True);(dest/'assets').mkdir(exist_ok=True)
for name in ['index.html','app.bundle.js','style.css','preview.css','embed.css']:
 shutil.copy2(source/name,dest/name)
# Bust old relative-runtime URLs when a previously cached iframe is revisited.
index=(dest/'index.html').read_text()
for name in ['app.bundle.js','embed.css']:
 revision=hashlib.sha256((dest/name).read_bytes()).hexdigest()[:12]
 index=index.replace('./'+name, './'+name+'?v='+revision)
(dest/'index.html').write_text(index)
# A fresh frame URL also bypasses previously cached card documents.
home=root/'index.html'
revision=hashlib.sha256(index.encode()).hexdigest()[:12]
home.write_text(re.sub(r'src="/card/\?embed=1(?:&(?:amp;)?v=[^"]*)?"',
                       'src="/card/?embed=1&amp;v='+revision+'"', home.read_text()))
config=json.loads((source/'card-config.json').read_text())
seen={}
for group in [config['assets'],config['back']['assets']]:
 for key,rel in group.items():
  if rel in seen:group[key]=seen[rel];continue
  src=source/rel
  if src.suffix=='.png':
   target=dest/'assets'/src.with_suffix('.webp').name
   # Keep text, line masks and alpha exact; lightly compress printed artwork.
   opts=['-lossless','-z','6'] if key in ['text','lineart'] else ['-q','90','-alpha_q','100','-m','5','-exact']
   subprocess.run(['cwebp','-quiet',*opts,str(src),'-o',str(target)],check=True)
  else:
   target=dest/'assets'/src.name;shutil.copy2(src,target)
  new='./assets/'+target.name;seen[rel]=new;group[key]=new
# Version every resource so an existing CDN/browser cache cannot mix releases.
for group in [config['assets'],config['back']['assets']]:
 for key,rel in group.items():
  digest=hashlib.sha256((dest/rel).read_bytes()).hexdigest()[:12]
  group[key]=rel+'?v='+digest
(dest/'card-config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
print('Published card runtime:',dest)
print('Runtime asset size:',round(sum(p.stat().st_size for p in (dest/'assets').iterdir())/1024/1024,2),'MiB')
