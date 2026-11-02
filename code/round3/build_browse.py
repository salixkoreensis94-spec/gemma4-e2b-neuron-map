"""Build compact numeric browsing data from the already-staged public records."""
import bisect,collections,gzip,json,math,struct
import hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT
def archive(p):
    if p.exists():
        target=(ROOT.parents[1] if ROOT.parent.name=='stage' else ROOT.parent)/'_superseded'/'round3_browse'/str(time.time_ns())/p.name
        target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);archive(p);p.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
TRANSPORT_LEVELS={'L14.bin.gz': 9, '48.json.gz': 9, '6d.json.gz': 9}
def gz(p,data):
    if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()
    assert len(data)<=25000000;p.parent.mkdir(parents=True,exist_ok=True);archive(p);p.write_bytes(gzip.compress(data,compresslevel=TRANSPORT_LEVELS.get(p.name,6),mtime=0))

TOKENS=49438284
WIDTHS=[6144]*15+[12288]*20
FREQUENCY=[0,.01,.05,.10,.25,.50,.75,1]
def quantile(values,q):
    at=(len(values)-1)*q;i=int(at);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(at-i)
def bin_of(edges,value):return bisect.bisect_right(edges[1:-1],value)
def main():
    stats=STAGE/'docs/map/data/round3/stats';dest=stats/'browse';catalogs=[];maximums=[];used=set();tokens=[]
    labels={};label_codes={None:0,'a':1,'b':2,'c':3,'u':4,'unknown':5}
    for row in (json.loads((stats/'ai_labels.json').read_text()) if (stats/'ai_labels.json').exists() else []):
        if row['kind']=='neuron':labels[(row['layer'],row['neuron'])]=label_codes[row['majority']]
    for layer,width in enumerate(WIDTHS):
        raw=gzip.decompress((stats/'catalog'/f'L{layer:02}.bin.gz').read_bytes());assert len(raw)==width*12;catalog=list(struct.iter_unpack('<IfBB2x',raw));catalogs.append(catalog);maximums.extend(x[1] for x in catalog)
        top=bytearray()
        for start in range(0,width,256):
            card=gzip.decompress((stats/'cards'/f'L{layer:02}_{start//256:02}.bin.gz').read_bytes())
            for off in range(0,len(card),580):
                ids=struct.unpack_from('<20I',card,off+338);counts=collections.Counter(ids);first={t:ids.index(t) for t in counts};fire=sorted(counts,key=lambda t:(-counts[t],first[t]))[:3];fire+= [0xffffffff]*(3-len(fire));lens=struct.unpack_from('<3I',card,off+418);row=fire+list(lens);top.extend(struct.pack('<6IB',*row,labels.get((layer,start+off//580),0)));used.update(t for t in row if t!=0xffffffff)
        assert len(top)==width*25;tokens.append(top);gz(dest/f'L{layer:02}.bin.gz',bytes(top))
    ordered=sorted(maximums);max_edges=sorted(set(quantile(ordered,q) for q in [0,.10,.25,.50,.75,.90,.99,1]));assert len(max_edges)>=2
    labels={};label_codes={None:0,'a':1,'b':2,'c':3,'u':4,'unknown':5}
    for row in (json.loads((stats/'ai_labels.json').read_text()) if (stats/'ai_labels.json').exists() else []):
        if row['kind']=='neuron':labels[(row['layer'],row['neuron'])]=label_codes[row['majority']]
    overview=bytearray();scale=math.log1p(ordered[-1]-ordered[0])
    for layer,catalog in enumerate(catalogs):
        for n,(positive,maximum,language,script) in enumerate(catalog):
            freq=positive/TOKENS;overview.extend(bytes([script,bin_of(FREQUENCY,freq),bin_of(max_edges,maximum),labels.get((layer,n),0),int(freq*255+.5),int(math.log1p(maximum-ordered[0])/scale*255+.5)]))
    assert len(overview)==337920*6;gz(dest/'overview.bin.gz',bytes(overview))
    vocab={}
    for p in (stats/'vocab').glob('*.json.gz'):
        for key,value in json.loads(gzip.decompress(p.read_bytes())).items():
            if int(key) in used:vocab[key]=value
    assert len(vocab)==len(used);gz(dest/'tokens.json.gz',vocab)
    vocab_sizes={int(p.name.split('.')[0]):p.stat().st_size for p in (stats/'vocab').glob('*.json.gz')};compact_bytes=(dest/'tokens.json.gz').stat().st_size;costs=[];list_cost=[];l0=[]
    for layer,catalog in enumerate(catalogs):
        for key in ['number','frequency','maximum']:
            for descending in [False,True]:
                ids=list(range(len(catalog)))
                if key=='number':ids.sort(reverse=descending)
                else:
                    col=0 if key=='frequency' else 1;ids.sort(key=lambda n:((-catalog[n][col] if descending else catalog[n][col]),n))
                if layer==0 and key=='maximum' and descending:l0=ids[:50]
                for start in range(0,len(ids),50):
                    blocks={t//1024 for n in ids[start:start+50] for t in struct.unpack_from('<6I',tokens[layer],n*25) if t!=0xffffffff};cost=sum(vocab_sizes[k] for k in blocks);costs.append(cost);list_cost.append(min(cost,compact_bytes))
    metadata=dict(neurons=337920,widths=WIDTHS,overview_record_bytes=6,overview_fields=['script_u8','frequency_bin_u8','maximum_bin_u8','AI_code_u8','frequency_brightness_u8','log_maximum_brightness_u8'],frequency_edges=FREQUENCY,maximum_edges=max_edges,maximum_brightness_min=ordered[0],maximum_brightness_max=ordered[-1],AI_labels=['none','a','b','c','u','unknown'],grid_columns=64,list_page_size=50,top_record_bytes=25,top_token_definition='Most common token IDs among 20 strongest contexts; ties by best context rank. No reservoir tokens in list.',top_fields=['strong_context_token_1','strong_context_token_2','strong_context_token_3','lens_token_1','lens_token_2','lens_token_3','AI_code_u8'],missing_token_id=0xffffffff,compact_vocab_candidate_bytes=compact_bytes,compact_vocab_bytes=None,vocab_shard_bytes=[vocab_sizes[i] for i in range(256)],overview_bytes=(dest/'overview.bin.gz').stat().st_size,brightness_definition='Script and sequential modes use fixed CSS-token colors without brightness mixing so the legend matches cells exactly; AI retains frequency brightness. Reduced pixels use the majority category, with smaller-code ties. Encoded brightness bytes are retained for format compatibility.',default_color='maximum',token_strategy='Existing vocabulary shards: the compact candidate was larger for every measured cold 50-row page. Cached strings cost zero.')
    archive(dest/'tokens.json.gz')
    write(dest/'metadata.json',metadata);write(dest/'validation.json',dict(L0_maximum_descending_first50=l0))
    report=dict(passed=True,neurons=337920,overview_raw_bytes=len(overview),overview_gzip_bytes=metadata['overview_bytes'],top3_gzip_bytes=sum((dest/f'L{l:02}.bin.gz').stat().st_size for l in range(35)),compact_vocab_tokens=len(used),compact_vocab_candidate_bytes=compact_bytes,compact_vocab_bytes=None,pages_measured=len(costs),page_rows=50,existing_shards_mean_bytes=sum(costs)/len(costs),existing_shards_min_bytes=min(costs),existing_shards_max_bytes=max(costs),adaptive_mean_bytes=sum(list_cost)/len(list_cost),adaptive_max_bytes=max(list_cost),compact_cheaper_pages=sum(c>compact_bytes for c in costs),grid_columns=64,context_text_printed=False);print(json.dumps(report))
if __name__=='__main__':main()
