"""Offline card, snippet, single-copy and static-load verification; no text output."""
import collections,gzip,json,math,re,struct
from pathlib import Path
from pii_patterns import categories
from snippet_policy import OFFENSE
ROOT=Path(__file__).resolve().parents[2]
def main():
    data=ROOT/'docs/map/data/round3';stats=data/'stats';compact=ROOT/'data/round3';opened=data/'snippets/open';manifest=json.loads((stats/'manifest.json').read_text());schema=json.loads((stats/'schema.json').read_text());assert manifest['neurons']==337920 and manifest['chunks']==263910 and manifest['snippet_status']=='complete'
    assert not (compact/'stats').exists() and not (compact/'snippets').exists()
    names={'m4.json','manifest.json','sources.json','schema.json','layers.json','round1.json','validation_examples.json','ai_labels.json','starting.json','README.md'}
    for p in compact.iterdir():
        assert p.is_file() and (p.name in names or re.fullmatch(r'm[23].*\.json',p.name)),p.name
        if p.suffix=='.json':assert p.read_bytes()==(stats/p.name).read_bytes(),p.name
    sources=json.loads((stats/'sources.json').read_text());assert len(sources)==6
    assert {s['display_license'] for s in sources}=={'CC BY 4.0','MIT','Apache-2.0 / public domain','ODC-By 1.0 + Common Crawl terms'}
    config=json.loads((ROOT/'docs/map/config.json').read_text());assert manifest['show_ai_labels']==config['show_ai_labels']
    label_bytes=(stats/'ai_labels.json').stat().st_size if config['show_ai_labels'] else 0
    refs={};text_files={};source_snippets=collections.Counter();maximum=0;unpacked_max=0;total=0
    for p in data.rglob('*'):
        if not p.is_file():continue
        size=p.stat().st_size;assert size<=25000000;total+=size;maximum=max(maximum,size)
        if p.suffix=='.gz':
            raw=gzip.decompress(p.read_bytes());unpacked_max=max(unpacked_max,len(raw));assert len(raw)<=25000000
            if p.name.endswith('.json.gz'):
                obj=json.loads(raw)
                if p.parent.name=='references':refs.update(obj)
                elif p.parent.parent==opened:
                    text_files[str(p.relative_to(opened))]=size;source_snippets[p.parent.name]+=len(obj)
                    for sid,entry in obj.items():
                        assert 0<entry['end']-entry['start']<=32 and not categories(entry['text'])
                        assert entry['offensive']==bool(OFFENSE.search(entry['text']))
                        assert sid.endswith(f"_{entry['start']}_{entry['end']}")
    # Reference geometry must resolve against exactly one stored snippet.
    entries_cache={}
    for key,ref in refs.items():
        name=ref['file']
        if name not in entries_cache:entries_cache[name]=json.loads(gzip.decompress((opened/name).read_bytes()))
        entry=entries_cache[name][ref['id']];length=len(entry['text'].encode('utf-16-le'))//2
        cid,off=map(int,key.split(':'));assert entry['start']<=off<entry['end'] and 0<=ref['a']<ref['b']<=length
    del entries_cache
    vocab_sizes={int(p.name.split('.')[0]):p.stat().st_size for p in (stats/'vocab').glob('*.json.gz')};pointer_sizes={int(p.name.split('.')[0]):p.stat().st_size for p in (stats/'pointers').glob('*.json.gz')};ref_sizes={int(p.name.split('.')[0]):p.stat().st_size for p in (opened/'references').glob('*.json.gz')};pointers={}
    for p in (stats/'pointers').glob('*.json.gz'):pointers.update(json.loads(gzip.decompress(p.read_bytes())))
    initial=['index.html','decoder.js','reader.js','browse.js','app.js','app.css','config.json','data/round3/stats/schema.json','data/round3/stats/manifest.json','data/round3/stats/layers.json','data/round3/stats/sources.json','data/round3/stats/round1.json','data/round3/stats/starting.json'];first=sum((ROOT/'docs/map'/p).stat().st_size for p in initial)
    deep=0;deep_card=None;count=0;referenced_slots=0;hidden_slots=0
    for p in (stats/'cards').glob('*.bin.gz'):
        raw=gzip.decompress(p.read_bytes());assert len(raw)%580==0
        for at in range(0,len(raw),580):
            count+=1
            ids=struct.unpack_from('<10I',raw,at+78)+struct.unpack_from('<20I',raw,at+338)+struct.unpack_from('<10I',raw,at+418)+struct.unpack_from('<10I',raw,at+498);chunks=struct.unpack_from('<20I',raw,at+138);offsets=struct.unpack_from('<20H',raw,at+218)
            cost=(stats/'catalog'/f'L{int(p.name[1:3]):02}.bin.gz').stat().st_size+label_bytes+p.stat().st_size+sum(vocab_sizes[k] for k in {t//1024 for t in ids})+sum(pointer_sizes[k] for k in {cid//256 for cid in chunks});ref_blocks=set();snippet_files=set()
            for rank,(cid,off) in enumerate(zip(chunks,offsets)):
                pointer=pointers[str(cid)];assert off<pointer[3]
                if not sources[pointer[0]]['open']:continue
                ref=refs.get(f'{cid}:{off}')
                if ref:referenced_slots+=1
                if rank<5:
                    ref_blocks.add(cid//256)
                    if ref:snippet_files.add(ref['file'])
            cost+=sum(ref_sizes[k] for k in ref_blocks)+sum(text_files[k] for k in snippet_files)
            if cost>deep:deep=cost;deep_card=dict(shard=p.name,row=at//580)
    coverage=json.loads((opened/'coverage.json').read_text());assert count==337920 and referenced_slots==coverage['published_references'];assert sum(source_snippets.values())==coverage['published_unique_snippets'];assert manifest['snippets']['context_limit']==20
    assert first<=2000000 and first+deep<=2000000
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=400000000
    for name in ['index.html','decoder.js','reader.js','browse.js','app.js','app.css']:
        value=(ROOT/'docs/map'/name).read_text();assert not re.search(r'https?://',value);assert not re.search(r'(?:src|href)=["\x27]/',value)
    assert not (data/'snippets/web').exists()
    for p in (ROOT/'preregistration/round3').rglob('*'):
        if p.is_file():assert not re.search(r'[/]Users[/]|[/]Volumes[/]|[/]home[/]|\bo[m]en\b',p.read_text(),re.I)
    labels=stats/'ai_labels.json'
    if labels.exists():
        rows=json.loads(labels.read_text());assert len(rows)==1820 and sum(r['kind']=='neuron' for r in rows)==1400
    for p in (ROOT/'code').rglob('*.py'):compile(p.read_text(),str(p.relative_to(ROOT)),'exec')
    import html,hashlib
    source=(ROOT/'RESULTS.md').read_text();r2=source.index('## Round 2:');r3=source.index('## Round 3:');copy=(ROOT/'docs/map/results.html').read_text();assert source[r3:] in html.unescape(copy) and source[r2:r3] in html.unescape(copy)
    assert 'results.html#round-3-a-full-neuron-map' in (ROOT/'docs/map/app.js').read_text()
    assert 'results.html#round-2-residual-sensors' in (ROOT/'docs/map/app.js').read_text()
    from check_browse import verify
    browse=verify()
    print(json.dumps(dict(**browse,passed=True,neurons=count,large_data_copies=1,published_snippets=sum(source_snippets.values()),published_references=referenced_slots,initial_load_bytes=first,worst_direct_card_load_bytes=first+deep,worst_direct_card=deep_card,pages_data_bytes=total,max_file_bytes=maximum,max_inflated_shard_bytes=unpacked_max,relative_urls_only=True,AI_labels_enabled=config['show_ai_labels'],AI_label_bytes=label_bytes,snippet_status='complete',context_text_printed=False)))
if __name__=='__main__':main()
