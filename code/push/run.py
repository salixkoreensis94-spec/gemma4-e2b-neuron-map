import hashlib,json,time,os
import numpy as np
import torch
from diagnose_numerics import ExactEngine
from generate_engine import OUT,ROOT,gpu_guard
from measures import measure
from measures_extra import extra
from diagnose_checks import register_problem
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main():
    assert json.loads((OUT/'engine_validation.json').read_text())['gate_passed']
    arms=json.loads((OUT/'arms.json').read_text());spec=json.loads((OUT/'directions.json').read_text());scenarios=json.loads((OUT/'scenarios.json').read_text())
    paths=[OUT/n for n in ['RULE.md','RULE.sha256','pairs_source.json','pairs.json','arms.json','scenarios.json','engine_validation.json','directions.json']]+list((OUT/'vectors').glob('*'))+[p for p in (OUT/'tools').glob('*.py') if p.name not in ['report.py','verify.py']]+[OUT/'inputs'/n for n in ['token_piece_hex.json','register_banmal_mean.gguf','rope_cpu_op.json','exp_cpu_corrections.bin']]+[ROOT/'inputs'/n for n in ['residual_norms.json','control_directions.f32','gelu_table.f16','gemma-4-E2B-it-Q4_K_M.gguf']]
    fingerprints={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(paths)};path=OUT/'logs/run_fingerprints.json'
    if path.exists():assert json.loads(path.read_text())==fingerprints,'resume inputs changed'
    else:
        with path.open('x') as f:json.dump(fingerprints,f,indent=2)
    prior=json.loads((OUT/'engine_validation.json').read_text())['seconds']+spec['seconds'];start=time.monotonic();gpu_guard();e=ExactEngine();e.fma=True;base={}
    vectors={name:{int(l):torch.from_numpy(np.load(OUT/t['path'],allow_pickle=False)).to('cuda') for l,t in d['tensors'].items()} for name,d in spec['directions'].items()}
    for arm in arms:
        for dose in arm['doses']:
            name=f"{arm['id']}_d{dose:g}";path=OUT/'checkpoints'/(name+'.json')
            if path.exists():
                assert path.with_suffix('.sha256').read_text().split()[0]==sha(path)
                data=json.loads(path.read_text());assert data['fingerprints']==fingerprints
                if arm['kind']=='none':base={r['scenario']:r for r in data['rows']}
                continue
            assert prior+time.monotonic()-start<3.75*3600,'budget stop'
            gpu_guard();push={} if arm['kind']=='none' else {l:v*dose for l,v in vectors[arm['id']].items()}
            rows,seconds=e.generate([s['token_ids'] for s in scenarios],push)
            for r,s in zip(rows,scenarios):
                r.update(arm=arm['id'],kind=arm['kind'],dose=dose,scenario=s['id'],target=arm['target'],context_register=s['context_register'],is_target_side=arm['target'] is not None and s['context_register']!=arm['target'])
                if arm['kind']=='none':base[s['id']]=dict(r)
                r.update(measure(r,base[s['id']],s));r.update(extra(r['reply'],base[s['id']]['reply']))
                if arm['target'] is not None:r['app_target_problem']=register_problem(r['reply'],'banmal' if arm['target']=='banmal' else 'haeyo',s['user'])
                r['target_dominant']=r[(arm['target'] or s['target'])+'_share']>.5
            data=dict(arm=arm['id'],dose=dose,seconds=seconds,rows=rows,vector_norms={str(l):float(torch.linalg.vector_norm(v)) for l,v in push.items()},fingerprints=fingerprints)
            with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            with path.with_suffix('.sha256').open('x') as f:f.write(sha(path)+'  '+path.name+'\n')
            target=[r for r in rows if r['is_target_side']]
            progress=dict(time=time.time(),arm=arm['id'],dose=dose,seconds=seconds,flips=sum(r['flip'] for r in target),friend_flips=sum(r['flip'] and r['friend_largest'] for r in target),pid=os.getpid())
            with (OUT/'logs/progress.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True)
    with (OUT/'logs/complete.json').open('x') as f:json.dump(dict(conditions=sum(len(a['doses']) for a in arms),generation_seconds=sum(json.loads(p.read_text())['seconds'] for p in (OUT/'checkpoints').glob('*.json')),run_wall_seconds=time.monotonic()-start,prior_validation_extraction_seconds=prior,finished_utc=time.time()),f,indent=2)
    print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
