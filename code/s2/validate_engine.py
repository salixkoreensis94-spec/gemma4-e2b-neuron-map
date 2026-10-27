from config import MODEL
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from engine import Engine,OUT,ROOT,MODEL,save_json

def comparison(a,b):
    a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
    at=np.lexsort((np.arange(a.size),-a))[:10]
    bt=np.lexsort((np.arange(b.size),-b))[:10]
    return dict(max_abs_diff=float(np.abs(a-b).max()),mean_abs_diff=float(np.abs(a-b).mean()),
                top1_equal=bool(at[0]==bt[0]),top10_overlap=int(len(set(at)&set(bt))),
                gpu_top10=at.tolist(),cpu_top10=bt.tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precision',default='highest');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();start=time.monotonic()
    engine=Engine(args.precision)
    prompts=json.loads((ROOT/'inputs/prompts_rendered.json').read_text())
    rows=[];layers=[]
    for i,p in enumerate(prompts[:1] if args.smoke else prompts):
        z,c=engine.forward(p['token_ids'],capture=i==0)
        ref=np.fromfile(OUT/f'reference/s1_{i}.f32',dtype='<f4')
        row=dict(id=i,**comparison(z.cpu().numpy(),ref));rows.append(row)
        print('S1',i,row,flush=True)
        if i==0:
            for l,h in enumerate(c['outputs']):
                ref=np.fromfile(OUT/f'reference/s1_0_layer_{l}.f32',dtype='<f4').reshape(h.shape)
                diff=h.cpu().numpy()-ref
                layers.append(dict(layer=l,max_abs_diff=float(np.abs(diff).max()),rms_diff=float(np.sqrt(np.mean(diff**2))),reference_rms=float(np.sqrt(np.mean(ref**2)))))
            print('layer differences',layers,flush=True)
    if args.smoke:
        save_json(OUT/f'logs/engine_smoke_{args.precision}.json',dict(rows=rows,layers=layers));return
    frames=json.loads((OUT/'frames.json').read_text());cache=engine.make_cache(frames)
    frame_rows=[]
    for i,f in enumerate(frames):
        frame_rows.append(dict(id=i,**comparison(cache['base'][i].cpu().numpy(),np.fromfile(OUT/f'reference/frame_{i}.f32',dtype='<f4'))))
    zero=[];pushed=[]
    for l,n in [(0,100),(1,100),(14,100),(15,100),(26,6893),(27,1939),(28,3794),(34,6703)]:
        h=engine.tail(cache,l,torch.zeros(1,1536,device='cuda'))
        zz=engine.head(h)
        zero.append(dict(layer=l,max_abs_diff=float((zz-cache['base']).abs().max().item()),
                         top1_equal=int((zz.argmax(-1)==cache['base'].argmax(-1)).sum().item())))
        v=engine.directions(l,[n])[0]
        for d in (-.5,.5):
            tail=engine.head(engine.tail(cache,l,(v*d)[None]))
            full,_=engine.forward(frames[0]['token_ids'],l,v*d,'last')
            one=dict(layer=l,neuron=n,dose=d,cached_vs_full=comparison(tail[0].cpu().numpy(),full.cpu().numpy()))
            if l in (1,26,27,28,34):
                allpos,_=engine.forward(frames[0]['token_ids'],l,v*d,'all')
                ref=np.fromfile(OUT/f'reference/push_{l}_{n}_{"pos" if d>0 else "neg"}.f32',dtype='<f4')
                one['allpos_vs_S1_CPU']=comparison(allpos.cpu().numpy(),ref)
            pushed.append(one)
    passed=all(r['top1_equal'] for r in rows) and np.mean([r['top10_overlap'] for r in rows])>=9
    extra=all(r['top1_equal']==12 and r['max_abs_diff']<.02 for r in zero) and all(r['cached_vs_full']['top1_equal'] for r in pushed)
    result=dict(schema='S2-engine-gate-v1',engine='minimal PyTorch Gemma4 ISWA, GGUF dequantized f32',
                precision=args.precision,torch_version=torch.__version__,numpy_version=np.__version__,
                device=torch.cuda.get_device_name(),model_sha256=hashlib.file_digest(MODEL.open('rb'),'sha256').hexdigest(),
                s1_prompts=rows,per_layer_prompt0=layers,max_abs_diff=max(r['max_abs_diff'] for r in rows),
                top1_agreement=sum(r['top1_equal'] for r in rows),top1_denominator=len(rows),
                mean_top10_overlap=float(np.mean([r['top10_overlap'] for r in rows])),
                gate_passed=bool(passed),additional_checks_passed=bool(extra),
                frames=frame_rows,zero_push_cached=zero,push_checks=pushed,
                seconds=time.monotonic()-start,
                insertion='l_out: after post-FFN norm + residual, PLE + residual, layer_output_scale; matches S1 cvec')
    result['implementation_sha256']={name:hashlib.file_digest((OUT/'tools'/name).open('rb'),'sha256').hexdigest() for name in ('engine.py','kernels.py')}
    result['gelu_table_sha256']=hashlib.file_digest((ROOT/'inputs/gelu_table.f16').open('rb'),'sha256').hexdigest()
    result['numerical_conventions']='S1 CPU short-batch Q8_K/bf16 activation rounding, fp16 KV, ARM fp16 attention accumulation, exact S1 GELU LUT; weights dequantized f32'
    save_json(OUT/'engine_validation.json',result)
    print('GATE',passed,'ADDITIONAL',extra,'max_abs_diff',result['max_abs_diff'],flush=True)
    if not(passed and extra):raise SystemExit(2)

if __name__=='__main__':main()
