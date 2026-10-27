"""Build content-only artifact and a normal static HTML page from identical body."""
import json,shutil,time
from pathlib import Path
REL=Path(__file__).resolve().parents[1]
S4=REL
D=REL/'data'
archive_root=REL.parents[1]/'_superseded' if REL.parent.name=='release' else REL.parent/'_superseded'
archive=archive_root/'page_builds'/str(time.time_ns())
for previous in [REL/'artifact_body.html',REL/'docs/index.html']:
 if previous.exists():
  archive.mkdir(parents=True,exist_ok=True);shutil.move(previous,archive/previous.name)
files={'map':'lens_map.json','cards':'neuron_cards.json','replies':'replies.json','scenarios':'scenarios.json','s3':'s3_conditions.json','s3b':'s3b_conditions.json','s6':'s6_conditions.json','s6all':'s6_all_conditions.json','human':'human_review_counts.json','layer':'s6_style_layer.json'}
data={k:json.loads((D/v).read_text()) for k,v in files.items()}
encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
body='<title>Gemma 4 뉴런 지도</title>\n<style>\n'+(REL/'code/page.css').read_text()+'\n</style>\n<main id="map-root"></main>\n<script>\nconst DATA = '+encoded+';\n'+(REL/'code/page.js').read_text()+'\n</script>\n'
assert len(body.encode())<5_000_000
(REL/'artifact_body.html').write_text(body)
(REL/'docs').mkdir(exist_ok=True)
(REL/'docs/index.html').write_text('<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n'+body+'</body></html>\n')
print('artifact bytes',len(body.encode()))
