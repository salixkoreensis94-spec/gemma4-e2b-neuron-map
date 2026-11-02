"""Verify every browse entry against the catalog and original public card rows."""
import bisect,collections,gzip,json,math,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE_FILES=['index.html','decoder.js','reader.js','browse.js','app.js','app.css','config.json','data/round3/stats/schema.json','data/round3/stats/manifest.json','data/round3/stats/layers.json','data/round3/stats/sources.json','data/round3/stats/round1.json','data/round3/stats/starting.json']
def verify():
    stats=ROOT/'docs/map/data/round3/stats';dest=stats/'browse';meta=json.loads((dest/'metadata.json').read_text());raw=gzip.decompress((dest/'overview.bin.gz').read_bytes());assert len(raw)==337920*6 and meta['overview_record_bytes']==6 and meta['top_record_bytes']==25
    codes={'a':1,'b':2,'c':3,'u':4,'unknown':5};labels={(r['layer'],r['neuron']):codes[r['majority']] for r in (json.loads((stats/'ai_labels.json').read_text()) if (stats/'ai_labels.json').exists() else []) if r['kind']=='neuron'};at=0;l0=None;page_max=0;sum_page=0;pages=0;max_catalog=0
    bins=lambda edges,v:bisect.bisect_right(edges[1:-1],v)
    for layer,width in enumerate(meta['widths']):
        cat_file=stats/'catalog'/f'L{layer:02}.bin.gz';cat=list(struct.iter_unpack('<IfBB2x',gzip.decompress(cat_file.read_bytes())));top_file=dest/f'L{layer:02}.bin.gz';tops=gzip.decompress(top_file.read_bytes());assert len(tops)==width*25;max_catalog=max(max_catalog,cat_file.stat().st_size)
        for n,(positive,maximum,language,script) in enumerate(cat):
            f=positive/49438284;expected=(script,bins(meta['frequency_edges'],f),bins(meta['maximum_edges'],maximum),labels.get((layer,n),0),int(f*255+.5),int(math.log1p(maximum-meta['maximum_brightness_min'])/math.log1p(meta['maximum_brightness_max']-meta['maximum_brightness_min'])*255+.5));assert tuple(raw[at:at+6])==expected;at+=6;assert tops[n*25+24]==expected[3]
        for start in range(0,width,256):
            cards=gzip.decompress((stats/'cards'/f'L{layer:02}_{start//256:02}.bin.gz').read_bytes())
            for pos in range(0,len(cards),580):
                n=start+pos//580;ids=struct.unpack_from('<20I',cards,pos+338);counts=collections.Counter(ids);fire=sorted(counts,key=lambda t:(-counts[t],ids.index(t)))[:3];fire+=[0xffffffff]*(3-len(fire));lens=struct.unpack_from('<3I',cards,pos+418);assert tuple(fire)+lens==struct.unpack_from('<6I',tops,n*25)
        for key in ['number','frequency','maximum']:
            for desc in [False,True]:
                ids=list(range(width))
                if key=='number':ids.sort(reverse=desc)
                else:
                    col=0 if key=='frequency' else 1;ids.sort(key=lambda n:((-cat[n][col] if desc else cat[n][col]),n))
                if layer==0 and key=='maximum' and desc:l0=ids[:50]
                for start in range(0,width,50):
                    blocks={t//1024 for n in ids[start:start+50] for t in struct.unpack_from('<6I',tops,n*25) if t!=0xffffffff};cost=sum(meta['vocab_shard_bytes'][b] for b in blocks);page_max=max(page_max,cat_file.stat().st_size+top_file.stat().st_size+cost);sum_page+=cost;pages+=1
    assert at==len(raw);assert l0==json.loads((dest/'validation.json').read_text())['L0_maximum_descending_first50'];assert not (dest/'tokens.json.gz').exists()
    first=sum((ROOT/'docs/map'/p).stat().st_size for p in BASE_FILES);ready=first+(dest/'metadata.json').stat().st_size+(dest/'overview.bin.gz').stat().st_size;grid=ready+max_catalog;listing=first+(dest/'metadata.json').stat().st_size+page_max
    assert ready<=2000000 and grid<=2000000 and listing<=2000000
    return dict(browse_verified=True,overview_entries=337920,overview_catalog_matches=True,top3_card_matches=True,L0_sorted_page_verified=True,overview_ready_bytes=ready,layer_grid_ready_max_bytes=grid,layer_list_ready_max_bytes=listing,browse_gzip_bytes=sum(p.stat().st_size for p in dest.glob('*.gz')),browse_rows_per_page=50,grid_columns=64,token_pages_measured=pages,token_page_mean_bytes=sum_page/pages,token_strategy='existing vocabulary shards',browser_viewport_measured=False)
if __name__=='__main__':print(json.dumps(dict(passed=True,**verify(),context_text_printed=False)))
