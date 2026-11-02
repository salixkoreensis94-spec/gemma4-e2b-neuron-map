"""Rebuild static assets and compact metadata; large data remains in Pages only."""
import argparse,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def archive(p):
    if p.exists():
        target=(ROOT.parents[1] if ROOT.parent.name=='stage' else ROOT.parent)/'_superseded'/'round3_page'/str(time.time_ns())/p.name
        target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
def report_copy():
    import html
    value=(ROOT/'RESULTS.md').read_text();start=value.index('## Round 2:');middle=value.index('## Round 3:');sections=[('round-3-a-full-neuron-map',value[middle:]),('round-2-residual-sensors',value[start:middle])]
    assert all('```' not in body and not any(line.startswith('>') for line in body.splitlines()) for _,body in sections)
    output='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>연구 결과 / Research results</title><link rel="stylesheet" href="app.css"><main><p><a href="index.html">지도로 돌아가기 / Back to the map</a></p><h1>연구 결과 원문 (영어) / Research report</h1><p>기존 RESULTS.md의 2·3차 결과를 그대로 옮겼습니다. / Unchanged Round 2 and Round 3 sections from RESULTS.md.</p>'
    for anchor,body in sections:
        output+='<section id="'+anchor+'"><pre class="report-copy" style="white-space:pre-wrap;overflow-wrap:anywhere;font:inherit">'+html.escape(body)+'</pre></section>'
    output+='</main></html>\n';out=ROOT/'docs/map/results.html';archive(out);out.write_text(output)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exclude-ai-labels',action='store_true');a=ap.parse_args();out=ROOT/'docs/map';data=out/'data/round3/stats';compact=ROOT/'data/round3';assert (data/'cards').is_dir();compact.mkdir(parents=True,exist_ok=True)
    if a.exclude_ai_labels:
        archive(data/'ai_labels.json');archive(compact/'ai_labels.json')
    config=json.loads((ROOT/'code/round3/map-config.json').read_text());config['show_ai_labels']=config['show_ai_labels'] and not a.exclude_ai_labels
    manifest=data/'manifest.json';metadata=json.loads(manifest.read_text());metadata['show_ai_labels']=config['show_ai_labels'];archive(manifest);manifest.write_text(json.dumps(metadata,ensure_ascii=False,separators=(',',':'))+'\n')
    if a.exclude_ai_labels and (data/'browse').exists():
        import gzip
        for p in (data/'browse').glob('*.bin.gz'):
            raw=bytearray(gzip.decompress(p.read_bytes()));stride,offset=(6,3) if p.name=='overview.bin.gz' else (25,24)
            raw[offset::stride]=bytes(len(raw[offset::stride]));archive(p);p.write_bytes(gzip.compress(bytes(raw),mtime=0))
    for p in data.glob('*.json'):
        target=compact/p.name;archive(target);shutil.copy2(p,target)
    for name in ['decoder.js','reader.js','browse.js','app.js','app.css']:
        p=out/name;archive(p);shutil.copy2(ROOT/'code/round3'/name,p)
    archive(out/'index.html');shutil.copy2(ROOT/'code/round3/map.html',out/'index.html')
    config=json.loads((ROOT/'code/round3/map-config.json').read_text());config['show_ai_labels']=config['show_ai_labels'] and not a.exclude_ai_labels
    archive(out/'config.json');(out/'config.json').write_text(json.dumps(config)+'\n')
    report_copy()
    print(json.dumps(dict(passed=True,large_data_copies=1,ai_labels_enabled=config['show_ai_labels'])))
if __name__=='__main__':main()
