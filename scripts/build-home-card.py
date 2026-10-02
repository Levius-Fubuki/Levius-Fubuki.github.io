"""Publish the card runtime with immutable URLs and resilient delivery."""
from pathlib import Path
import json, shutil, subprocess, hashlib, re
root=Path(__file__).resolve().parents[1]
source=root/'card-studio/levius-id/web';dest=root/'card'
dest.mkdir(exist_ok=True);(dest/'assets').mkdir(exist_ok=True)
subprocess.run(['npx','--yes','esbuild@0.25.0',str(source/'app.js'),'--bundle','--format=esm','--minify','--target=es2020','--outfile='+str(dest/'app.bundle.js')],check=True)
for name in ['style.css','preview.css','embed.css']:
 shutil.copy2(source/name,dest/name)
def versioned_copy(path):
 digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
 versioned=path.with_name(path.stem+'.'+digest+path.suffix)
 shutil.copy2(path,versioned)
 return './'+str(versioned.relative_to(dest))
config=json.loads((source/'card-config.json').read_text())
seen={}
for group in [config['assets'],config['back']['assets']]:
 for key,rel in group.items():
  if rel in seen:group[key]=seen[rel];continue
  src=source/rel
  if src.suffix=='.png':
   target=dest/'assets'/src.with_suffix('.webp').name
   opts=['-lossless','-z','6'] if key in ['text','lineart'] else ['-q','90','-alpha_q','100','-m','5','-exact']
   subprocess.run(['cwebp','-quiet',*opts,str(src),'-o',str(target)],check=True)
  else:
   target=dest/'assets'/src.name;shutil.copy2(src,target)
  new=versioned_copy(target);seen[rel]=new;group[key]=new
config_json=json.dumps(config,ensure_ascii=False,indent=2)+'\n'
(dest/'card-config.json').write_text(config_json)
index=(source/'index.html').read_text()
index=re.sub(r'\s*<script type="module"[^>]*src="\./app.bundle.js[^>]*></script>', '',index)
# Blocking CSS must not prevent the recovery bootstrap from executing.
# All card styles/font data are local and small enough to ship with the document.
for name in ['style.css','preview.css','embed.css']:
 css=(dest/name).read_text()
 index=re.sub(r'<link rel="stylesheet" href="\./'+re.escape(name)+r'"\s*/>',
              lambda _: '<style data-card-style="'+name+'">'+css+'</style>',index)
module=versioned_copy(dest/'app.bundle.js')
bootstrap=(root/'scripts/card-runtime/asset-loader.mjs').read_text().replace('export ', '')+'\n'+(root/'scripts/card-runtime/bootstrap.js').read_text()
index=index.replace('</body>', '<script id="card-runtime-config" type="application/json">'+config_json.replace('<','\\u003c')+'</script>\n<script type="module" data-card-module="'+module+'">\n'+bootstrap+'\n</script>\n</body>')
(dest/'index.html').write_text(index)
home=root/'index.html';revision=hashlib.sha256(index.encode()).hexdigest()[:12]
home.write_text(re.sub(r'src="/card/\?embed=1(?:&(?:amp;)?v=[^"]*)?"', 'src="/card/?embed=1&amp;v='+revision+'"',home.read_text()))
print('Published immutable card runtime:', module)
print('Runtime assets:',len(seen))
