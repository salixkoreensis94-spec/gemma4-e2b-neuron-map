from config import MODEL
import hashlib,json,time,sys,os
import torch
from generate_engine import GenerateEngine,OUT,ROOT,gpu_guard
from diagnose_numerics import ExactEngine
from measures import measure
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main():
    validation=json.loads((OUT/'engine_validation.json').read_text());assert validation['gate_passed']
    scenarios=json.loads((OUT/'scenarios.json').read_text());arms=json.loads((OUT/'arms.json').read_text())
    paths=[OUT/'RULE.md',OUT/'arms.json',OUT/'scenarios.json',OUT/'engine_validation.json',OUT/'inputs/register_banmal_mean.gguf',OUT/'inputs/token_piece_hex.json',OUT/'inputs/rope_cpu_op.json',OUT/'inputs/exp_cpu_corrections.bin',ROOT/'inputs/gelu_table.f16',ROOT/'inputs/control_directions.f32',ROOT/'inputs/residual_norms.json',MODEL]+[OUT/'tools'/n for n in ['run_arms.py','generate_engine.py','diagnose_numerics.py','exact_linear.py','s2_engine.py','kernels.py','measures.py','diagnose_checks.py']]
    fingerprints={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)):digest(p) for p in paths}
    manifest=OUT/'logs/run_fingerprints.json'
    if manifest.exists():assert json.loads(manifest.read_text())==fingerprints,'resume fingerprints changed'
    else:
        with manifest.open('x') as f:json.dump(fingerprints,f,indent=2)
    e=ExactEngine();e.fma=True;start=time.monotonic();elapsed=0
    saved=list((OUT/'checkpoints').glob('*.json'))
    elapsed=sum(json.loads(p.read_text())['seconds'] for p in saved)
    base={}
    for ai,arm in enumerate(arms):
        for dose in arm['doses']:
            name=f"{arm['id']}_d{dose:g}";path=OUT/'checkpoints'/(name+'.json')
            if path.exists():
                assert path.with_suffix('.sha256').read_text().split()[0]==digest(path),'checkpoint checksum'
                old=json.loads(path.read_text());assert old['fingerprints']==fingerprints
                if arm['kind']=='none':base={r['scenario']:r for r in old['rows']}
                continue
            if elapsed+time.monotonic()-start>5.0*3600:raise RuntimeError('compute budget checkpoint stop')
            gpu_guard();push=e.vectors(arm,dose)
            norms={str(l):float(torch.linalg.vector_norm(v).item()) for l,v in push.items()}
            # batch all 20, each retains its own left-padded KV and absolute positions.
            rows,seconds=e.generate([s['token_ids'] for s in scenarios],push)
            for row,s in zip(rows,scenarios):
                row.update(arm=arm['id'],kind=arm['kind'],dose=dose,scenario=s['id'],target=arm['target'],context_register=s['context_register'])
                if arm['kind']=='none':base[s['id']]=dict(row)
                row.update(measure(row,base[s['id']],s))
            data=dict(arm=arm['id'],dose=dose,seconds=seconds,rows=rows,vector_norms=norms,fingerprints=fingerprints)
            with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            with path.with_suffix('.sha256').open('x') as f:f.write(digest(path)+'  '+path.name+'\n')
            with (OUT/'logs/progress.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),arm=arm['id'],dose=dose,seconds=seconds,arm_index=ai,total_arms=len(arms)))+'\n')
            print(name,'seconds',round(seconds,2),'target_flips',sum(r['flip'] for r in rows if r['context_register']!=arm['target']),flush=True)
            (OUT/'logs/job_status.json').write_text(json.dumps(dict(stage='arms',time=time.time(),arm=arm['id'],dose=dose,arm_index=ai,total_arms=len(arms),pid=os.getpid()))+'\n')
            if ai==0:
                estimate=seconds*(1+sum(len(a['doses']) for a in arms[1:]))
                with (OUT/'logs/benchmark.json').open('x') as f:json.dump(dict(seconds_20_scenarios=seconds,estimated_all_seconds=estimate,dose_checkpoints=289),f,indent=2)
                if estimate>5.0*3600:raise RuntimeError('benchmark exceeds budget; optimize before arms')
    print('ALL COMPLETE',time.monotonic()-start,flush=True)
    (OUT/'logs/job_status.json').write_text(json.dumps(dict(stage='complete',time=time.time(),pid=os.getpid()))+'\n')
if __name__=='__main__':
    try:main()
    except Exception as err:
        (OUT/'logs/job_status.json').write_text(json.dumps(dict(stage='failed',time=time.time(),error=repr(err),pid=os.getpid()))+'\n')
        raise
