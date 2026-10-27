import hashlib,json,time
import torch
from diagnose_numerics import ExactEngine
from generate_engine import OUT,gpu_guard
def main():
    gpu_guard();e=ExactEngine();e.fma=True
    prompts=json.loads((OUT/'validation_prompts.json').read_text());cpu=json.loads((OUT/'reference/cpu_greedy.json').read_text())
    rows=[];start=time.monotonic()
    for p,c in zip(prompts,cpu):
        r,seconds=e.generate([p['token_ids']]);a=c['token_ids'];b=r[0]['token_ids']
        rows.append(dict(id=p['id'],equal=a==b,first_divergence=next((i for i in range(max(len(a),len(b))) if a[i:i+1]!=b[i:i+1]),None),cpu_token_ids=a,gpu=r[0],seconds=seconds))
        print('validation',p['id'],a==b,round(seconds,2),flush=True)
    batch,seconds=e.generate([p['token_ids'] for p in prompts]);consistent=[r['gpu']['token_ids']==b['token_ids'] for r,b in zip(rows,batch)]
    zero,_=e.generate([prompts[0]['token_ids']],{27:torch.zeros(e.D,device='cuda')})
    z=zero[0]['token_ids']==rows[0]['gpu']['token_ids'];exact=sum(r['equal'] for r in rows)
    data=dict(rows=rows,exact_replies=exact,denominator=10,batch_consistency=consistent,zero_push_equal=z,gate_passed=exact>=9 and all(consistent) and z,seconds=time.monotonic()-start,cpu_reference='Reused S3/reference/cpu_greedy.json; no new reference inference',implementation_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (OUT/'tools').glob('*.py')})
    with (OUT/'engine_validation.json').open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    assert data['gate_passed'], 'validation gate failed'
    print('VALIDATED',exact,flush=True)
if __name__=='__main__':main()
