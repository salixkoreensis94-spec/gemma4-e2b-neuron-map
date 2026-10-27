"""Independent artifact, coverage, selection, reconstruction and raw-token audit."""
import collections,csv,gzip,json,statistics
import numpy as np
from b_common import O,R,STYLES,LAYERS,DOSES,CORE,RULE_HASH,load,save,sha,freeze_check
from style_classifier import classify

def main():
 freeze_check();selection=load(O/'selection.json');plan=load(O/'dev_conditions.json')
 phases={p:{x.stem:load(x) for x in sorted((O/'checkpoints'/p).glob('*.json'))} for p in ['dev','heldout']}
 assert len(phases['dev'])==len(plan['conditions'])*2+5
 assert len(phases['heldout'])==5+3*sum(x['selected'] is not None for x in selection['styles'].values())
 pieces=[bytes.fromhex(x) for x in load(O/'inputs/token_piece_hex.json')]
 rows_checked=0
 for phase,by in phases.items():
  ss=load(O/f'scenarios_{phase}.json');ids={s['id'] for s in ss};n=len(ss)
  for key,data in by.items():
   path=O/f'checkpoints/{phase}/{key}.json';assert PathHash(path)==sha(path)
   assert data['rule_sha256']==RULE_HASH
   for p,h in data['fingerprint'].items():assert sha(O/p)==h,p
   assert len(data['rows'])==n and {r['scenario'] for r in data['rows']}==ids
   for r in data['rows']:
    tokens=r['token_ids'];raw=b''.join(pieces[t] for t in tokens if t not in (1,106))
    assert raw.hex()==r['raw_hex'] and raw.decode('utf8','replace')==r['reply']
    assert r['length']==sum(t not in (1,106) for t in tokens) and len(tokens)<=96
    assert r['truncated']==(tokens[-1] not in (1,106))
    c=classify(r['reply']);assert r['dominant_style']==c['dominant'] and r['style_counts']==c['counts'] and r['style_shares']==c['shares']
    core=sum(t['label']=='haera' and t['tail'].endswith(CORE) for t in c['tails'])
    assert r['haera_core']==dict(count=core,segments=c['segments'],share=core/max(1,c['segments']))
    baseline=next(x for x in by['none']['rows'] if x['scenario']==r['scenario'])
    assert r['length_ratio']==r['length']/max(1,baseline['length'])
    assert r['second_person_delta']==r['second_person']-baseline['second_person']
    rows_checked+=1
 # Independent frozen threshold and lexicographic selection calculation.
 independent=[]
 for c in plan['conditions']:
  rr=phases['dev'][c['key']]['rows'];cc=phases['dev'][c['control_key']]['rows'];bb=phases['dev']['none']['rows'];style=c['style']
  dom=sum(r['dominant_style']==style for r in rr);bdom=sum(r['dominant_style']==style for r in bb);cdom=sum(r['dominant_style']==style for r in cc)
  h=sum(r['hangul_share']>=.95 for r in rr);d=sum(r['distinct4']>=.9 for r in rr);l=sum(.5<=r['length_ratio']<=2 for r in rr)
  delta=sum(r['second_person_delta'] for r in rr)/20
  passed=dom>=14 and bdom<=3 and cdom<=3 and min(h,d,l)>=18 and delta<=.5
  independent.append(dict(key=c['key'],style=style,passed=passed,rank=(-dom,-min(h,d,l),c['dose'],c['layer'],c['name'])))
  a=next(a for a in selection['dev_summaries'] if a['key']==c['key'])
  assert (a['target_dominant'],a['none_dominant'],a['control_dominant'],a['hangul_pass'],a['distinct4_pass'],a['length_pass'],a['second_person_delta'],a['passed'])==(dom,bdom,cdom,h,d,l,delta,passed)
 for s in STYLES:
  yes=sorted([a for a in independent if a['style']==s and a['passed']],key=lambda a:a['rank'])
  no=sorted([a for a in independent if a['style']==s and not a['passed']],key=lambda a:a['rank'])
  x=selection['styles'][s]
  assert (x['selected']['key'] if x['selected'] else None)==(yes[0]['key'] if yes else None)
  assert (x['near_miss']['key'] if x['near_miss'] else None)==(no[0]['key'] if no else None)
 # Reconstruct every contrastive mean from saved last-token activations.
 captures={}
 for p in sorted((O/'extraction').glob('*.json')):
  m=load(p);a=np.load(O/m['activations_path']);assert sha(O/m['activations_path'])==m['activations_sha256']
  for t,v in zip(m['tasks'],a):captures[(t['pair'],t['side'])]=v
 assert len(captures)==256
 pairs=[p for p in load(O/'pairs.json') if p['split']=='build'];norms=load(R/'inputs/residual_norms.json')['layers'];maxerr=0.
 for s in STYLES:
  pp=[p for p in pairs if p['style']==s];assert len(pp)==32
  difference=np.stack([captures[(p['id'],'positive')]-captures[(p['id'],'negative')] for p in pp])
  for li,l in enumerate(LAYERS):
   name=f'{s}.dir.L{l:02}';raw=difference[:,li].mean(0,dtype=np.float32)
   assert np.array_equal(raw,np.load(O/f'vectors/{name}.raw.npy'))
   expected=raw/np.linalg.norm(raw)*norms[l]['median'];actual=np.load(O/f'vectors/{name}.npy')
   err=float(abs(actual-expected).max());maxerr=max(maxerr,err);assert err<1e-4
 control_errors=[];gpu_control_errors=[]
 for name,v in plan['vectors'].items():
  assert sha(O/v['path'])==v['sha256'] and sha(O/v['control_path'])==v['control_sha256']
  target=np.load(O/v['path']);control=np.load(O/v['control_path'])
  err=float(abs(np.linalg.norm(target)-np.linalg.norm(control))/np.linalg.norm(target));control_errors.append(err);assert err<5e-7
  gpu_err=abs(v['norm']-v['control_norm'])/v['norm'];gpu_control_errors.append(gpu_err);assert gpu_err<3e-7
 # Reconstruct full lens top40 from every saved score, and neuron p95 admission.
 scores=[]
 for l in range(35):
  p=O/f'lens/layer_{l:02}.npz';m=load(p.with_suffix('.json'));assert sha(p)==m['sha256']
  arr=np.load(p)['scores'];assert arr.shape[1]==4 and len(arr)==m['neurons']
  for n,sc in enumerate(arr):scores.append((l,n,sc))
 lens_meta=load(O/'lens/top40.json');top=lens_meta['styles'];ns=load(O/'neuron_selection.json')
 direct_error=max(x['max_error'] for x in lens_meta['direct_verification']);boundary_gaps={}
 assert len(ns['candidates'])==160
 for si,s in enumerate(STYLES):
  ordered=sorted(scores,key=lambda t:(-t[2][si],t[0],t[1]));expected=ordered[:40]
  boundary_gaps[s]=float(ordered[39][2][si]-ordered[40][2][si])
  assert boundary_gaps[s]>2*direct_error,('numerically close lens cutoff',s,boundary_gaps[s],direct_error)
  assert [(t[0],t[1]) for t in expected]==[(t['layer'],t['neuron']) for t in top[s]]
  for c in [c for c in ns['candidates'] if c['style']==s]:
   l=c['layer'];m=load(O/f'screen/L{l:02}_controls.json');controls=np.load(O/m['arrays_path'])['shifts'];assert controls.shape==(512,12,4)
   threshold=float(np.quantile(controls[:,:,si].mean(1,dtype=np.float64),.95,method='linear'))
   assert threshold==c['control_p95'] and c['beats_p95']==(c['mean_shift']>threshold)
  candidates=sorted([c for c in ns['candidates'] if c['style']==s and c['beats_p95']],key=lambda c:(-c['mean_shift'],c['layer'],c['neuron']))[:2]
  assert candidates==ns['admitted'][s]
 for p in (O/'screen').glob('*.json'):
  m=load(p);assert PathHash(p)==sha(p) and sha(O/m['arrays_path'])==m['arrays_sha256']
  assert m['zero_cached_parity']['top1_equal']==12 and m['zero_cached_parity']['max_logit_delta']<.02
  for name,h in m['fingerprint'].items():assert sha(O/name)==h
 # Export and review coverage checks.
 with (O/'map/style_layer.tsv').open() as f:
  rd=csv.DictReader(f,delimiter='\t');assert 'haera_core' in rd.fieldnames;maps=list(rd)
 assert len(maps)==100
 assert {(m['style'],int(m['layer']),float(m['dose'])) for m in maps}=={(s,l,d) for s in STYLES for l in LAYERS for d in DOSES}
 cos=load(O/'map/cosines.json')
 for l,entry in cos['layers'].items():
  uv={s:np.load(O/f'vectors/{s}.dir.L{int(l):02}.npy').astype(np.float64) for s in STYLES}
  uv={s:v/np.linalg.norm(v) for s,v in uv.items()}
  for a in STYLES:
   for b in STYLES:assert abs(entry['pairwise'][a][b]-float(uv[a]@uv[b]))<1e-6
  for s,refs in entry['references'].items():
   assert (refs['app'] is not None)==(int(l) in (8,14))
   assert (refs['N1939'] is not None)==(int(l)==27)
   assert (refs['S3b_friend_last'] is not None)==(int(l)==27)
   assert refs['N6893'] is None
 for phase,file in [('dev','review_dev.md'),('heldout','review_heldout.md')]:
  txt=(O/file).read_text();assert 'haera_core' in txt
  for s in load(O/f'scenarios_{phase}.json'):assert txt.count('### '+s['id']+' ')==4,s['id']
 with gzip.open(O/'generations.tsv.gz','rt') as f:assert sum(1 for _ in csv.DictReader(f,delimiter='\t'))==rows_checked
 assert 'haera_core' in (O/'summary.md').read_text()
 audit=dict(rule_sha256=RULE_HASH,full_dev_coverage=True,full_heldout_coverage=True,raw_reply_rows_checked=rows_checked,
            independently_recomputed_selection=True,all20_direction_raw_means_reconstructed=True,max_vector_coordinate_error=maxerr,
            max_control_relative_norm_error_numpy=max(control_errors),numpy_norm_check_tolerance=5e-7,
            max_control_relative_norm_error_at_gpu_construction=max(gpu_control_errors),gpu_construction_norm_tolerance=3e-7,
            all35_lens_layers_and_top40_verified=True,all160_screen_candidates_and_p95_verified=True,
            every_screen_layer_zero_cache_gate_verified=True,
            lens_top40_boundary_gaps=boundary_gaps,lens_max_direct_float32_error=direct_error,
            haera_core_descriptive_only=True,review_coverage=True,map_rows=100)
 save(O/'logs/phase_b_output_audit.json',audit)
 # Runtime completion marker and open runtime logs are mirrored but not sealed here.
 excluded={'SHA256SUMS.txt','logs/DONE_B','logs/driver_b.log','logs/b_verify.log'}
 files=[p for p in sorted(O.rglob('*')) if p.is_file() and str(p.relative_to(O)) not in excluded
        and not any(x in p.relative_to(O).parts for x in ['_superseded','__pycache__','inputs'])]
 with (O/'SHA256SUMS.txt').open('x') as f:
  for p in files:f.write(sha(p)+'  '+str(p.relative_to(O))+'\n')
 print(json.dumps(dict(**audit,sealed_files=len(files)),indent=2))
def PathHash(p):return __import__('pathlib').Path(str(p)+'.sha256').read_text().strip()
if __name__=='__main__':main()
