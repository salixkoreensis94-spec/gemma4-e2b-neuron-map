"""Derive plain-reader metadata and search additions from public card data only."""
import bisect,collections,gzip,json,struct,unicodedata
from pathlib import Path
from build_browse import ROOT,WIDTHS,quantile,write,gz
def normal(s):return unicodedata.normalize('NFKC',s).strip().lower()
def bucket(s):
    h=2166136261
    for b in s.encode():h=((h^b)*16777619)&0xffffffff
    return h&255
def strong(ids):
    c=collections.Counter(ids);return sorted(c,key=lambda t:(-c[t],ids.index(t)))
def contains_letter(piece):
    if piece=='[withheld token]' or chr(0xfffd) in piece:return False
    for ch in piece:
        name=unicodedata.name(ch,'')
        if unicodedata.category(ch).startswith('L') and ch not in '\u115f\u1160\u3164\uffa0' and ('LATIN' in name or 'HANGUL' in name or name.startswith(('CJK UNIFIED IDEOGRAPH','CJK COMPATIBILITY IDEOGRAPH'))):return True
    return False
def main(starting_only=False):
    stats=ROOT/'docs/map/data/round3/stats';browse=stats/'browse';meta=json.loads((browse/'metadata.json').read_text());layers=json.loads((stats/'layers.json').read_text());vocab={}
    for p in (stats/'vocab').glob('*.json.gz'):vocab.update(json.loads(gzip.decompress(p.read_bytes())))
    raw=gzip.decompress((browse/'overview.bin.gz').read_bytes());maximums=[];freq=[];catalogs=[]
    for l in range(35):
        cat=list(struct.iter_unpack('<IfBB2x',gzip.decompress((stats/'catalog'/f'L{l:02}.bin.gz').read_bytes())));catalogs.append(cat);maximums.extend(r[1] for r in cat);freq.extend(r[0]/49438284 for r in cat)
    ordered=sorted(maximums);percentiles=[0,.1,.25,.5,.75,.9,.99,1];derived=[quantile(ordered,p) for p in percentiles];assert derived==meta['maximum_edges']
    max_counts=[0]*7;frequency_counts=[0]*7
    for i in range(337920):max_counts[raw[i*6+2]]+=1;frequency_counts[raw[i*6+1]]+=1
    assert max_counts==[33792,50688,84480,84480,50688,30412,3380]
    assert frequency_counts[4:6]==[153531,153489]
    meta.update(maximum_percentiles=percentiles,maximum_bin_counts=max_counts,frequency_bin_counts=frequency_counts,positive_definition='activation > 0',strongest_median_layer=max(range(35),key=lambda l:layers[l]['maximum']['p50']),weakest_median_layer=min(range(35),key=lambda l:layers[l]['maximum']['p50']))
    write(browse/'metadata.json',meta)
    # Include only already-public single token pieces, using the original search normalization.
    updates=[collections.defaultdict(list) for _ in range(256)];labels=json.loads((stats/'ai_labels.json').read_text());a={(r['layer'],r['neuron']) for r in labels if r['kind']=='neuron' and r['majority']=='a'};candidates=[];special=None;base=0
    for l,width in enumerate(WIDTHS):
        for start in range(0,width,256):
            cards=gzip.decompress((stats/'cards'/f'L{l:02}_{start//256:02}.bin.gz').read_bytes())
            for off in range(0,len(cards),580):
                n=start+off//580;ids=struct.unpack_from('<20I',cards,off+338);maximum=struct.unpack_from('<f',cards,off+12)[0]
                if not starting_only:
                    for key in {normal(vocab[str(t)]) for t in ids}-{'','[withheld token]'}:updates[bucket(key)][key].append(base+n)
                if (l,n) in a or (l,n)==(27,1939):
                    chunks=struct.unpack_from('<20I',cards,off+138);positions=struct.unpack_from('<20H',cards,off+218);c=collections.Counter(ids);entry=dict(layer=l,neuron=n,maximum=maximum,token_ids=strong(ids)[:3],counts=[c[t] for t in strong(ids)[:3]],top5=[[chunks[i],positions[i]] for i in range(5)],top1_token_id=ids[0],top1_contains_letter=contains_letter(vocab[str(ids[0])]),letter_contexts=sum(contains_letter(vocab[str(t)]) for t in ids))
                    if (l,n)==(27,1939):special=entry
                    elif (l,n) in a:candidates.append(entry)
        base+=width
    before=sum(p.stat().st_size for p in (stats/'index').glob('*.gz'));new_postings=0
    if not starting_only:
        for b in range(256):
            p=stats/'index'/f'{b:02x}.json.gz';index=json.loads(gzip.decompress(p.read_bytes()))
            for key,ids in updates[b].items():
                old=index.get(key,[]);merged=sorted(set(old)|set(ids));new_postings+=len(merged)-len(old);index[key]=merged
            gz(p,index)
    # Check availability with reference metadata only; no snippet text is read.
    wanted=collections.defaultdict(set)
    for row in candidates:
        for cid,off in row['top5']:wanted[cid//256].add(f'{cid}:{off}')
    available=set();opened=ROOT/'docs/map/data/round3/snippets/open'
    for block,keys in wanted.items():
        refs=json.loads(gzip.decompress((opened/'references'/f'{block:04}.json.gz').read_bytes()));available.update(keys&refs.keys())
    eligible=[r for r in candidates if r['top1_contains_letter'] and r['letter_contexts']>=10 and sum(f'{cid}:{off}' in available for cid,off in r['top5'])>=3];chosen=[];fallbacks=[];empty_bands=[]
    for band in range(7):
        if not any(r['layer']//5==band for r in eligible):empty_bands.append([band*5,band*5+4])
        for step in range(7):
            target=(band+step)%7;used={(r['layer'],r['neuron']) for r in chosen}
            options=[r for r in eligible if r['layer']//5==target and (r['layer'],r['neuron']) not in used]
            if options:
                row=dict(min(options,key=lambda r:(-r['maximum'],r['layer'],r['neuron'])));row['requested_band']=[band*5,band*5+4];row['selected_band']=[target*5,target*5+4];chosen.append(row)
                if step:fallbacks.append(dict(requested_band=row['requested_band'],selected_band=row['selected_band'],layer=row['layer'],neuron=row['neuron'],wrapped=target<band))
                break
        else:raise AssertionError('Fewer than seven eligible starting neurons')
    assert len(chosen)==7 and special is not None
    chosen.append(special)
    for row in chosen:
        row['tokens']=[vocab[str(t)] for t in row['token_ids']];row.pop('top5')
    rule='Seven M2-majority-a neurons with >=3 available open snippets among first 5 contexts, a Hangul/Latin/Han letter in the strongest (rank-1) context token, and letter-bearing tokens at >=10 of 20 context positions. For each five-layer band in ascending order choose the highest maximum among unused candidates; if empty, advance to the next band cyclically, taking its next unused candidate. Ties layer/neuron ascending. Plus L27 N1939 as the published exception. No new semantic titles.'
    write(stats/'starting.json',dict(rule=rule,items=chosen,empty_bands=empty_bands,fallbacks=fallbacks))
    for name in ['schema.json','manifest.json']:
        value=json.loads((stats/name).read_text())
        if name=='schema.json':value['token_search']='NFKC, trim, lowercase exact pieces over positive-reservoir top10, lens top10 and all distinct strongest-context top20 token IDs; FNV1a UTF8 low-byte sharding.'
        else:value['browse'].update(list_tokens='top20 context-token counts, descending; ties best rank',search_context_tokens=20)
        write(stats/name,value)
    compact=ROOT/'data/round3'
    for name in ['schema.json','manifest.json','starting.json']:write(compact/name,json.loads((stats/name).read_text()))
    after=sum(p.stat().st_size for p in (stats/'index').glob('*.gz'))
    print(json.dumps(dict(passed=True,maximum_edges=derived,maximum_bin_counts=max_counts,frequency_bin_counts=frequency_counts,frequency_middle_share=sum(frequency_counts[4:6])/337920,AI_label_neurons=1400,index_before_bytes=before,index_after_bytes=after,index_added_postings=new_postings,context_token_limit=20,starting_ids=[[r['layer'],r['neuron']] for r in chosen],starting_eligible=len(eligible),starting_rule=rule,empty_bands=empty_bands,fallbacks=fallbacks,lazy_catalog_max_bytes=max((stats/'catalog'/f'L{l:02}.bin.gz').stat().st_size for l in range(35)),context_text_printed=False)))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--starting-only',action='store_true');args=parser.parse_args();main(args.starting_only)
