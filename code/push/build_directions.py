import hashlib,json,time
import numpy as np
import torch
from diagnose_numerics import ExactEngine
from generate_engine import ROOT,OUT,gpu_guard
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def write(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def unit(v):return v/torch.linalg.vector_norm(v)
def main():
    assert json.loads((OUT/'engine_validation.json').read_text())['gate_passed']
    gpu_guard();start=time.monotonic();e=ExactEngine();e.fma=True
    pairs=json.loads((OUT/'pairs.json').read_text());tasks=[]
    for p in pairs:
        for side in ['positive','negative']:tasks.append(dict(pair=p['id'],family=p['family'],side=side,ids=p[side+'_ids'],start=p[side+'_start']))
    captured={};all_means=[];all_lasts=[]
    for at in range(0,len(tasks),8):
        gpu_guard();batch=tasks[at:at+8];b=len(batch);length=max(len(x['ids']) for x in batch)
        ids=np.zeros((b,length),dtype=np.int64);valid=torch.zeros((b,length),device='cuda',dtype=torch.bool);pos=torch.zeros((b,length),device='cuda');mask=torch.zeros((b,length),device='cuda')
        for i,x in enumerate(batch):
            n=len(x['ids']);ids[i,-n:]=x['ids'];valid[i,-n:]=True;pos[i,-n:]=torch.arange(n,device='cuda');mask[i,length-n+x['start']:]=1
        e.capture_reply_mask=mask;e.step(ids,pos,valid,{},{});a=e.reply_capture.clone();z=e.last_capture.clone()
        for x,m,last in zip(batch,a,z):captured[(x['pair'],x['side'])]=(m,last)
        all_means.extend(a.cpu().numpy());all_lasts.extend(z.cpu().numpy())
        print('extracted',at+b,'/',len(tasks),flush=True)
    e.capture_reply_mask=None
    differences={family:[captured[(p['id'],'positive')][0]-captured[(p['id'],'negative')][0] for p in pairs if p['family']==family] for family in ['register','second_person']}
    fmean=unit(torch.stack(differences['register']).mean(0));you_raw=torch.stack(differences['second_person']).mean(0);you=unit(you_raw)
    flast=unit(torch.stack([captured[(p['id'],'positive')][1]-captured[(p['id'],'negative')][1] for p in pairs if p['family']=='register']).mean(0))
    n=unit(e.directions(27,[1939])[0]);dot=torch.dot(n,you);removed=n-dot*you;minus=unit(removed)
    mixa=.4*n+.6*minus;mixb=.4*n+.6*fmean
    v={'N1939':{27:e.directions(27,[1939])[0]},'N6893':{26:e.directions(26,[6893])[0]},'app':{l:e.app[l] for l in range(4,15)}}
    for name,a in [('minus_you',minus),('friend_mean',fmean),('friend_last',flast),('mix_minus_you',unit(mixa)),('mix_friend',unit(mixb)),('reverse_friend_mean',-fmean),('reverse_friend_last',-flast)]:v[name]={27:a*e.norms[27]['median']}
    arms=json.loads((OUT/'arms.json').read_text());meta={}
    for arm in arms:
        name=arm['id']
        if name=='none':continue
        if arm['kind']=='control':
            original=arm['paired'];v[name]={}
            for l,source in v[original].items():
                r=unit(e.directions(l,[318 if original=='N6893' else 317],controls=True)[0]);sign=-1 if original.startswith('reverse_') else 1
                v[name][l]=r*torch.linalg.vector_norm(source)*sign
        tensors={}
        for l,a in v[name].items():
            path=OUT/'vectors'/f'{name}_L{l}.npy'
            with path.open('xb') as f:np.save(f,a.cpu().numpy().astype('<f4'),allow_pickle=False)
            tensors[str(l)]=dict(path=str(path.relative_to(OUT)),sha256=sha(path),norm=float(torch.linalg.vector_norm(a)))
        meta[name]=dict(**arm,tensors=tensors)
    for name,a in [('second_person_unit',you),('register_mean_raw',torch.stack(differences['register']).mean(0)),('second_person_raw',you_raw)]:
        path=OUT/'vectors'/f'{name}.npy'
        with path.open('xb') as f:np.save(f,a.cpu().numpy().astype('<f4'),allow_pickle=False)
    with (OUT/'vectors/reply_activations.npz').open('xb') as f:np.savez_compressed(f,means=np.array(all_means,dtype='<f4'),lasts=np.array(all_lasts,dtype='<f4'))
    write('directions.json',dict(construction='Frozen README_ko.md initial construction; see RULE.md and pairs_source.json',layer_indexing='zero based',rule_sha256=sha(OUT/'RULE.md'),pairs_sha256=sha(OUT/'pairs.json'),source_pairs_sha256=sha(OUT/'pairs_source.json'),median_norms={str(l):e.norms[l]['median'] for l in [26,27]},directions=meta,n_dot_second_person=float(dot),removed_component_norm=float(abs(dot)),remaining_fraction=float(torch.linalg.vector_norm(removed)),orthogonality_after=float(torch.dot(minus,you)),cosines=dict(n_friend_mean=float(torch.dot(n,fmean)),n_friend_last=float(torch.dot(n,flast)),friend_mean_last=float(torch.dot(fmean,flast)),you_friend_mean=float(torch.dot(you,fmean))),mixture_unit_component_coefficients=dict(mix_minus_you=[.4/float(torch.linalg.vector_norm(mixa)),.6/float(torch.linalg.vector_norm(mixa))],mix_friend=[.4/float(torch.linalg.vector_norm(mixb)),.6/float(torch.linalg.vector_norm(mixb))]),extraction_tasks=[{k:x[k] for k in ['pair','family','side','start']}|{'tokens':len(x['ids'])} for x in tasks],auxiliary_vector_hashes={p.name:sha(p) for p in sorted((OUT/'vectors').glob('*'))},seconds=time.monotonic()-start))
    print('BUILT',len(meta),'directions',round(time.monotonic()-start,2),'seconds',flush=True)
if __name__=='__main__':main()
