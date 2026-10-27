"""Complete frozen Phase B GPU pipeline; resumable by condition and extraction chunk."""
import gc,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from b_common import *
from style_classifier import classify
sys.path.insert(0,str(R/'S3b/tools'))
import generate_engine
generate_engine.OUT=O
from diagnose_numerics import ExactEngine
from measures import measure
from measures_extra import extra

def unit(v):return v/torch.linalg.vector_norm(v)
def guard():generate_engine.gpu_guard()
def fingerprint():
 return {str(p.relative_to(O)):sha(p) for p in [O/'freeze.json',O/'tokenized_inputs.json',*[O/'tools'/n for n in ['b_common.py','b_lens.py','b_compute.py','style_classifier.py']]]}
SIG=None
def checkpoint(path,data):
 data=dict(**data,fingerprint=SIG,rule_sha256=RULE_HASH)
 save(path,data)
 with Path(str(path)+'.sha256').open('x') as f:f.write(sha(path)+'\n')
def resume(path):
 freeze_check()
 if not path.exists():return None
 assert Path(str(path)+'.sha256').read_text().strip()==sha(path)
 data=load(path);assert data['fingerprint']==SIG and data['rule_sha256']==RULE_HASH,path
 return data

def token_audit():
 tasks=load(O/'tokenized_inputs.json');tokens={t['id']:t['token_ids'] for t in tasks}
 assert len(tasks)==len(tokens)==554
 for s in load(O/'scenarios_dev.json'):assert tokens[f'dev/{s["id"]}/none']==s['token_ids']
 for p in load(R/'S3b/validation_prompts.json'):assert tokens[f'validation/{p["id"]}']==p['token_ids']
 for p in load(O/'pairs.json'):
  if p['split']!='build':continue
  pre=tokens[f'pair/{p["id"]}/prefix']
  for side in ['positive','negative']:
   ids=tokens[f'pair/{p["id"]}/{side}'];assert ids[:len(pre)]==pre and len(ids)>len(pre)
 assert max(len(t['token_ids']) for t in tasks if not t['id'].startswith('pair/'))+96<=512
 path=O/'logs/tokenization_audit_b.json'
 if not path.exists():save(path,dict(tasks=len(tasks),dev_equal=20,validation_equal=10,prefix_boundaries_verified=256,max_prompt_tokens=max(map(len,tokens.values())),tokenized_sha256=sha(O/'tokenized_inputs.json')))
 return tokens

def extraction(e,tokens):
 pairs=[p for p in load(O/'pairs.json') if p['split']=='build'];tasks=[dict(pair=p['id'],side=s,ids=tokens[f'pair/{p["id"]}/{s}']) for p in pairs for s in ['positive','negative']]
 captured={};st=time.monotonic()
 for at in range(0,len(tasks),8):
  path=O/f'extraction/chunk_{at:03}.json';old=resume(path)
  if old:
   arr=np.load(O/old['activations_path']);assert sha(O/old['activations_path'])==old['activations_sha256']
  else:
   guard();batch=tasks[at:at+8];b=len(batch);length=max(len(t['ids']) for t in batch)
   ids=np.zeros((b,length),dtype=np.int64);valid=torch.zeros((b,length),device='cuda',dtype=torch.bool);pos=torch.zeros((b,length),device='cuda')
   for i,t in enumerate(batch):
    n=len(t['ids']);ids[i,-n:]=t['ids'];valid[i,-n:]=True;pos[i,-n:]=torch.arange(n,device='cuda')
   e.trace=True;e.layer_trace={};e.step(ids,pos,valid,{},{})
   arr=np.stack([np.asarray(e.layer_trace[l],dtype='<f4') for l in LAYERS],axis=1)
   e.trace=False;e.layer_trace={}
   ap=path.with_suffix('.npy');ap.parent.mkdir(exist_ok=True)
   with ap.open('xb') as f:np.save(f,arr,allow_pickle=False)
   checkpoint(path,dict(tasks=[{k:v for k,v in t.items() if k!='ids'} for t in batch],layers=LAYERS,
                        activations_path=str(ap.relative_to(O)),activations_sha256=sha(ap)))
   progress('extraction_chunk',tasks=at+b,total=len(tasks))
  for t,a in zip(tasks[at:at+8],arr):captured[(t['pair'],t['side'])]=a
 vectors={}
 for style in STYLES:
  pp=[p for p in pairs if p['style']==style]
  diff=np.stack([captured[(p['id'],'positive')]-captured[(p['id'],'negative')] for p in pp])
  for li,l in enumerate(LAYERS):
   name=f'{style}.dir.L{l:02}';path=O/f'vectors/{name}.npy';raw=diff[:,li].mean(0,dtype=np.float32)
   v=unit(torch.from_numpy(raw.copy()).to('cuda'))*e.norms[l]['median']
   vectors[name]=dict(name=name,kind='direction',style=style,layer=l,path=str(path.relative_to(O)),norm=float(torch.linalg.vector_norm(v)))
   if not path.exists():
    path.parent.mkdir(exist_ok=True)
    with path.open('xb') as f:np.save(f,v.cpu().numpy().astype('<f4'),allow_pickle=False)
    with path.with_name(name+'.raw.npy').open('xb') as f:np.save(f,raw,allow_pickle=False)
   else:assert np.array_equal(np.load(path),v.cpu().numpy())
   vectors[name]['sha256']=sha(path)
 path=O/'directions_b.json'
 if not path.exists():checkpoint(path,dict(vectors=vectors,seconds=time.monotonic()-st,construction='Equal-pair final-reply-token style minus partner; 32 build pairs only; all five frozen layers'))
 progress('directions_complete',count=len(vectors),seconds=time.monotonic()-st)
 return vectors

def screen():
 output=O/'neuron_selection.json';old=resume(output)
 if old:return old['admitted']
 # The original S2 cached screen has a different linear/attention numerical path
 # from S3b's autoregressive CPU-exact engine. Keep the validated S2 path intact.
 import importlib.util
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 sk=module('s6_original_s2_kernels',R/'S2/tools/kernels.py')
 se=module('s6_original_s2_engine',R/'S2/tools/engine.py');se.kernels=sk
 validation=load(R/'S2/engine_validation.json')
 assert validation['gate_passed'] and validation['additional_checks_passed']
 for name,h in validation['implementation_sha256'].items():assert sha(R/'S2/tools'/name)==h
 guard();e=se.Engine('highest')
 top=load(O/'lens/top40.json')['styles'];frames=load(R/'S2/frames.json')
 guard();cache=e.make_cache(frames);base=cache['base'];lp0=F.log_softmax(base,dim=-1);ent0=-(lp0.exp()*lp0).sum(-1)
 groups=load(O/'token_groups.json')['groups'];partner=load(O/'token_group_plan.json')['partner']
 vocab=load(R/'inputs/vocabulary.json')
 import re
 plausible=np.array([bool(t['type']==1 and re.search('[가-힣]',t['string'].replace('▁',' ')) and re.fullmatch(r'[가-힣ᄀ-ᇿ㄰-㆏\s.!?~…]+',t['string'].replace('▁',' '))) for t in vocab])
 def statistic(z,s):return torch.logsumexp(z[:,groups[s]['screen_ids']],1)-torch.logsumexp(z[:,groups[partner[s]]['screen_ids']],1)
 br=torch.stack([statistic(base,s) for s in STYLES],1)
 def evaluate(push,l):
  h=e.tail(cache,l,push);zs=[];ks=[];es=[];ts=[]
  for at in range(0,len(h),96):
   z=e.head(h[at:at+96]);fi=torch.arange(at,at+len(z),device='cuda')%12
   lp=F.log_softmax(z,dim=-1);prob=lp.exp()
   zs.append((torch.stack([statistic(z,s) for s in STYLES],1)-br[fi]).reshape(-1,12,4))
   ks.append((prob*(lp-lp0[fi])).sum(-1));es.append(-(prob*lp).sum(-1)-ent0[fi]);ts.append(z.argmax(-1))
  shifts=torch.cat(zs).cpu().numpy();kl=torch.cat(ks).reshape(-1,12).cpu().numpy();entropy=torch.cat(es).reshape(-1,12).cpu().numpy();top1=torch.cat(ts).reshape(-1,12).cpu().numpy()
  assert np.isfinite(shifts).all() and np.isfinite(kl).all() and kl.min()>-1e-4
  return dict(shifts=shifts,kl=kl,entropy_change=entropy,top1=top1,plausible=plausible[top1])
 layers=sorted({x['layer'] for a in top.values() for x in a});summaries=[]
 for l in layers:
  guard();freeze_check();indices=sorted({x['neuron'] for a in top.values() for x in a if x['layer']==l})
  for controls,indices0 in [(True,list(range(512))),(False,indices)]:
   kind='controls' if controls else 'candidates';path=O/f'screen/L{l:02}_{kind}.json';old=resume(path)
   if old:continue
   arrays={k:[] for k in ['shifts','kl','entropy_change','top1','plausible']};st=time.monotonic()
   dirs=e.directions(l,indices0,controls=controls)
   zero=e.head(e.tail(cache,l,torch.zeros(1,e.D,device='cuda')))
   parity=dict(top1_equal=int((zero.argmax(-1)==base.argmax(-1)).sum()),max_logit_delta=float((zero-base).abs().max()))
   assert parity['top1_equal']==12 and parity['max_logit_delta']<.02,('S2 cached zero mismatch',l,parity)
   for at in range(0,len(dirs),8):
    if at%64==0:guard()
    r=evaluate(dirs[at:at+8]*.5,l)
    for k,v in r.items():arrays[k].append(v)
   ap=path.with_suffix('.npz');ap.parent.mkdir(exist_ok=True)
   with ap.open('xb') as f:np.savez_compressed(f,**{k:np.concatenate(v) for k,v in arrays.items()})
   checkpoint(path,dict(layer=l,kind=kind,indices=indices0,arrays_path=str(ap.relative_to(O)),arrays_sha256=sha(ap),seconds=time.monotonic()-st,zero_cached_parity=parity))
   progress('screen_layer_kind',layer=l,kind=kind,count=len(indices0),seconds=time.monotonic()-st,zero=parity)
  control=load(O/f'screen/L{l:02}_controls.json');cm=np.load(O/control['arrays_path'])['shifts'].mean(1,dtype=np.float64)
  candidate=load(O/f'screen/L{l:02}_candidates.json');ca=np.load(O/candidate['arrays_path']);means=ca['shifts'].mean(1,dtype=np.float64)
  for si,s in enumerate(STYLES):
   threshold=float(np.quantile(cm[:,si],.95,method='linear'))
   for j,n in enumerate(candidate['indices']):
    entry=next((x for x in top[s] if x['layer']==l and x['neuron']==n),None)
    if entry is None:continue
    summaries.append(dict(style=s,layer=l,neuron=n,lens_score=entry['scores'][s],mean_shift=float(means[j,si]),control_p95=threshold,
                          beats_p95=bool(means[j,si]>threshold),shifts=ca['shifts'][j,:,si].tolist(),kl=ca['kl'][j].tolist(),
                          entropy_change=ca['entropy_change'][j].tolist(),top1=ca['top1'][j].tolist(),plausible=ca['plausible'][j].tolist()))
 admitted={s:sorted([x for x in summaries if x['style']==s and x['beats_p95']],key=lambda x:(-x['mean_shift'],x['layer'],x['neuron']))[:2] for s in STYLES}
 checkpoint(output,dict(admitted=admitted,candidates=summaries,screen_group_ids={s:groups[s]['screen_ids'] for s in [*STYLES,'friend','haeyo']}))
 del cache,lp0,ent0,base;gc.collect();torch.cuda.empty_cache()
 progress('screen_complete',admitted={s:[(x['layer'],x['neuron']) for x in a] for s,a in admitted.items()})
 return admitted

def add_neurons(e,vectors,admitted):
 for s,rows in admitted.items():
  for r in rows:
   l,n=r['layer'],r['neuron'];name=f'{s}.neuron.L{l:02}.N{n:05}';v=e.directions(l,[n])[0];path=O/f'vectors/{name}.npy'
   if not path.exists():
    with path.open('xb') as f:np.save(f,v.cpu().numpy().astype('<f4'),allow_pickle=False)
   else:assert np.array_equal(np.load(path),v.cpu().numpy())
   vectors[name]=dict(name=name,kind='neuron',style=s,layer=l,neuron=n,path=str(path.relative_to(O)),norm=float(torch.linalg.vector_norm(v)),sha256=sha(path))
 return vectors

def rows_with_measures(raw,scenarios,base,condition):
 rows=[]
 for r,s in zip(raw,scenarios):
  r=dict(r);r.update(arm=condition['name'],kind=condition['kind'],dose=condition.get('dose',0),scenario=s['id'],context_register=s['context_register'],target_style=condition.get('style'))
  b=base[s['id']] if base else dict(r)
  r.update(measure(r,b,s));r.update(extra(r['reply'],b['reply']))
  c=classify(r['reply']);core=sum(t['label']=='haera' and t['tail'].endswith(CORE) for t in c['tails'])
  r.update(style_counts=c['counts'],style_shares=c['shares'],style_segments=c['segments'],style_tails=c['tails'],dominant_style=c['dominant'],
           haera_core=dict(count=core,segments=c['segments'],share=core/max(1,c['segments'])))
  rows.append(r)
 return rows

def generate_condition(e,tokens,phase,condition,base):
 key=condition['key'];path=O/f'checkpoints/{phase}/{key}.json';old=resume(path)
 if old:return old
 guard();scenarios=load(O/f'scenarios_{phase}.json');system=condition.get('prompt_style','none')
 prompts=[tokens[f'{phase}/{s["id"]}/{system}'] for s in scenarios]
 push={}
 if 'vector_path' in condition:
  v=torch.from_numpy(np.load(O/condition['vector_path']).copy()).to('cuda');push={condition['layer']:v*condition['dose']}
 start=time.monotonic();raw,seconds=e.generate(prompts,push)
 data=dict(condition=condition,rows=rows_with_measures(raw,scenarios,base,condition),seconds=seconds,
           vector_norms={str(l):float(torch.linalg.vector_norm(v)) for l,v in push.items()})
 checkpoint(path,data)
 progress('generation_condition',phase=phase,key=key,seconds=seconds,dominant=dict(__import__('collections').Counter(r['dominant_style'] for r in data['rows'])))
 return data

def controls(e,vectors):
 for name,v in vectors.items():
  l=v['layer'];source=torch.from_numpy(np.load(O/v['path'])).to('cuda');r=unit(e.directions(l,[317],controls=True)[0])*torch.linalg.vector_norm(source)
  path=O/f'vectors/random.{name}.npy'
  if not path.exists():
   with path.open('xb') as f:np.save(f,r.cpu().numpy().astype('<f4'),allow_pickle=False)
  else:assert np.array_equal(np.load(path),r.cpu().numpy())
  v['control_path']=str(path.relative_to(O));v['control_norm']=float(torch.linalg.vector_norm(r));v['control_sha256']=sha(path)
  assert abs(v['control_norm']-v['norm'])/v['norm']<3e-7

def grid(e,tokens,vectors):
 conditions=[]
 for name,v in vectors.items():
  for d in DOSES:
   c=dict(name=name,kind=v['kind'],style=v['style'],layer=v['layer'],dose=d,key=f'{name}.d{d:g}',vector_path=v['path'])
   c['control_key']=f'random.{name}.d{d:g}';conditions.append(c)
 if not (O/'dev_conditions.json').exists():checkpoint(O/'dev_conditions.json',dict(conditions=conditions,vectors=vectors))
 none=dict(name='none',kind='none',key='none');data=generate_condition(e,tokens,'dev',none,None);base={r['scenario']:r for r in data['rows']}
 # Benchmark a full condition before committing to the complete grid.
 if not (O/'logs/generation_benchmark.json').exists():save(O/'logs/generation_benchmark.json',dict(none_seconds=data['seconds'],max_grid_conditions=len(conditions)*2+5,estimated_dev_seconds=data['seconds']*(len(conditions)*2+5),heldout_reserve_seconds=600))
 for s in STYLES:generate_condition(e,tokens,'dev',dict(name=f'prompt.{s}',kind='prompt',style=s,prompt_style=s,key=f'prompt.{s}'),base)
 summaries=[]
 for c in conditions:
  a=generate_condition(e,tokens,'dev',c,base)
  cc=dict(name='random.'+c['name'],kind='control',style=c['style'],layer=c['layer'],dose=c['dose'],key=c['control_key'],vector_path=vectors[c['name']]['control_path'],paired_key=c['key'])
  b=generate_condition(e,tokens,'dev',cc,base)
  agg=aggregate(a['rows'],c['style']);ctrl=aggregate(b['rows'],c['style']);ba=aggregate(list(base.values()),c['style'])
  summaries.append(dict(**c,**agg,**decide(agg,ba,ctrl)))
 selected={}
 for s in STYLES:
  rr=[a for a in summaries if a['style']==s];passed=sorted([a for a in rr if a['passed']],key=rank);failed=sorted([a for a in rr if not a['passed']],key=rank)
  selected[s]=dict(status='knob' if passed else 'no knob',selected=passed[0] if passed else None,near_miss=failed[0] if failed else None)
 out=O/'selection.json'
 if not out.exists():checkpoint(out,dict(styles=selected,dev_summaries=summaries))
 else:assert load(out)['styles']==selected
 return selected

def heldout(e,tokens,vectors,selected):
 none=generate_condition(e,tokens,'heldout',dict(name='none',kind='none',key='none'),None);base={r['scenario']:r for r in none['rows']}
 for s in STYLES:
  generate_condition(e,tokens,'heldout',dict(name=f'prompt.{s}',kind='prompt',style=s,prompt_style=s,key=f'prompt.{s}'),base)
  c=selected[s]['selected']
  if c is None:continue
  plain={k:c[k] for k in ['name','kind','style','layer','dose','key','vector_path','control_key']}
  generate_condition(e,tokens,'heldout',plain,base)
  generate_condition(e,tokens,'heldout',dict(plain,name='prompt+'+c['name'],key='prompt+'+c['key'],prompt_style=s),base)
  generate_condition(e,tokens,'heldout',dict(name='random.'+c['name'],kind='control',style=s,layer=c['layer'],dose=c['dose'],key=c['control_key'],vector_path=vectors[c['name']]['control_path'],paired_key=c['key']),base)

def cosines(e,vectors):
 path=O/'map/cosines.json'
 if path.exists():return
 entries={name:unit(torch.from_numpy(np.load(O/v['path'])).to('cuda')) for name,v in vectors.items() if v['kind']=='direction'}
 result={}
 for l in LAYERS:
  vv={s:entries[f'{s}.dir.L{l:02}'] for s in STYLES}
  refs={}
  if l==27:
   refs['N1939']=unit(e.directions(27,[1939])[0]);refs['S3b_friend_last']=unit(torch.from_numpy(np.load(R/'S3b/vectors/friend_last_L27.npy')).to('cuda'))
  if l==26:refs['N6893']=unit(e.directions(26,[6893])[0])
  if l in e.app:refs['app']=unit(e.app[l])
  result[str(l)]=dict(pairwise={a:{b:float(torch.dot(vv[a],vv[b])) for b in STYLES} for a in STYLES},
                      references={s:{ref:float(torch.dot(v,refs[ref])) if ref in refs else None for ref in ['N1939','N6893','S3b_friend_last','app']} for s,v in vv.items()})
 save(path,dict(layers=result,missing='null means no same-layer reference; no layer26 style direction, so all N6893 comparisons are not applicable',reference_sha256={'S3b_friend_last':sha(R/'S3b/vectors/friend_last_L27.npy')}))

def main():
 global SIG
 freeze_check();SIG=fingerprint();tokens=token_audit();start=time.monotonic()
 from b_lens import run
 if not (O/'lens/top40.json').exists():run(guard)
 guard();e=ExactEngine();e.fma=True
 vectors=extraction(e,tokens)
 del e;gc.collect();torch.cuda.empty_cache()
 admitted=screen();gc.collect();torch.cuda.empty_cache()
 guard();e=ExactEngine();e.fma=True
 vectors=add_neurons(e,vectors,admitted);controls(e,vectors)
 cosines(e,vectors);selected=grid(e,tokens,vectors);heldout(e,tokens,vectors,selected)
 checkpoint(O/'logs/gpu_pipeline_complete.json',dict(seconds=time.monotonic()-start,selected={s:x['selected']['key'] if x['selected'] else None for s,x in selected.items()},vectors=len(vectors)))
 progress('gpu_pipeline_complete',seconds=time.monotonic()-start)

if __name__=='__main__':main()
