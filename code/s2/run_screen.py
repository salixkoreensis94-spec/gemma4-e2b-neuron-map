from config import MODEL
from config import NVIDIA_SMI
"""GPU pilot and resumable per-layer exhaustive screen. Never remove files."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from engine import Engine,OUT,ROOT,MODEL,save_json
from metrics import Metrics,GraphEvaluator

DOSES=(-.8,-.5,.5,.8)
HEADER=['layer','neuron','dose','mean_shift','positive_frames','negative_frames',
        'plausible_frames','mean_kl','shifts','kl','entropy_change','top1','plausible']

def now():return datetime.now(timezone.utc).isoformat()
def log(event,**data):
    obj=dict(time=now(),event=event,**data)
    with (OUT/'logs/progress.jsonl').open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def gpu_check(layer=None):
    raw=subprocess.check_output([NVIDIA_SMI,'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    used,total,util=[int(x.strip()) for x in raw.split(',')]
    reserved=torch.cuda.memory_reserved()/2**20 if torch.cuda.is_initialized() else 0
    external=max(0,used-reserved)
    log('gpu_check',layer=layer,total_used_mib=used,total_mib=total,own_reserved_mib=reserved,
        external_estimate_mib=external,utilization_percent=util,raw=raw)
    return external<=3072

def wait_free():
    while not gpu_check():
        log('pause_other_gpu_use',poll_seconds=180)
        time.sleep(180)
    log('gpu_available')

def load_runtime(precision):
    wait_free()
    e=Engine(precision);frames=json.loads((OUT/'frames.json').read_text())
    c=e.make_cache(frames);m=Metrics(e,frames,c)
    return e,c,m

def signature():
    paths=[OUT/'RULE.md',OUT/'frames.json',OUT/'engine_validation.json',MODEL,
           ROOT/'inputs/residual_norms.json',ROOT/'inputs/control_directions.f32',
           ROOT/'inputs/gelu_table.f16',OUT/'tools/engine.py',OUT/'tools/kernels.py',OUT/'tools/metrics.py',OUT/'tools/run_screen.py']
    return {(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)):sha(p) for p in paths}

def ensure_gate():
    v=json.loads((OUT/'engine_validation.json').read_text())
    assert v['gate_passed'] and v['additional_checks_passed'],'Engine validation gate must pass'
    for name,digest in v['implementation_sha256'].items():assert sha(OUT/'tools'/name)==digest,'Engine changed after gate'
    rulehash=(OUT/'RULE.md.sha256').read_text().split()[0]
    assert sha(OUT/'RULE.md')==rulehash,'Changed rule'
    return v['precision']

def benchmark(e,c,m,batch,repeats=3):
    rows=[]
    for l in (0,17,34):
        dirs=e.directions(l,list(range(batch)))
        evaluator=GraphEvaluator(e,c,m,l,batch)
        # Warm up final tensor shapes and metric operations.
        evaluator(dirs*.5);torch.cuda.synchronize()
        ts=[]
        for rep in range(repeats):
            st=time.monotonic()
            r=evaluator(dirs*.5)
            torch.cuda.synchronize();ts.append(time.monotonic()-st)
        row=dict(layer=l,batch=batch,frames=12,repetitions_seconds=ts,
                 median_seconds=float(np.median(ts)),seconds_per_direction=float(np.median(ts)/batch))
        rows.append(row);log('benchmark',**row)
        del evaluator;gc.collect();torch.cuda.empty_cache()
    xp=np.array([r['layer'] for r in rows]);yp=np.array([r['seconds_per_direction'] for r in rows])
    layers=[]
    for l,w in enumerate(e.widths):
        rate=float(np.interp(l,xp,yp))
        layers.append(dict(layer=l,neurons=w,controls=512,estimated_seconds=rate*(w+512)*4))
    estimate=sum(x['estimated_seconds'] for x in layers)
    result=dict(frozen_before_screen=now(),batch=batch,measurements=rows,layer_estimates=layers,
                screen_estimated_seconds=estimate,screen_estimated_hours=estimate/3600,
                allowance_validation_and_io_percent=25,total_estimated_hours=estimate*1.25/3600,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    return result

def serial(v):return json.dumps(v,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def write_batch(writer,l,indices,dose,r):
    for j,n in enumerate(indices):
        s=r['shifts'][j];kl=r['kl'][j];pl=r['plausible'][j]
        writer.writerow([l,int(n),dose,format(float(s.mean(dtype=np.float64)),'.10g'),
                         int((s>0).sum()),int((s<0).sum()),int(pl.sum()),format(float(kl.mean(dtype=np.float64)),'.10g'),
                         serial(s.tolist()),serial(kl.tolist()),serial(r['entropy_change'][j].tolist()),
                         serial(r['top1'][j].tolist()),serial(pl.astype(int).tolist())])

def run_layer(e,c,m,l,batch,sig):
    path=OUT/'layers'/f'layer_{l:02d}';path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():
        data=json.loads((path/'complete.json').read_text())
        assert data['signature']==sig,'Resume inputs changed'
        for name,digest in data['sha256'].items():assert sha(path/name)==digest
        log('skip_completed_layer',layer=l);return
    existing=list(path.iterdir())
    if existing:
        dest=OUT/'_superseded'/('incomplete_'+time.strftime('%Y%m%d_%H%M%S'))/path.name
        dest.mkdir(parents=True)
        for p in existing:p.rename(dest/p.name)
    st=time.monotonic()
    evaluator=GraphEvaluator(e,c,m,l,batch)
    files={}
    for control in (True,False):
        name='controls.tsv.gz' if control else 'screen.tsv.gz'
        width=512 if control else e.widths[l]
        assert width%batch==0
        dirs=e.directions(l,controls=control)
        with gzip.open(path/name,'wt',compresslevel=1,newline='') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(HEADER)
            for dose in DOSES:
                for at in range(0,width,batch):
                    indices=list(range(at,min(width,at+batch)))
                    r=evaluator(dirs[at:at+len(indices)]*dose)
                    write_batch(writer,l,indices,dose,r)
                log('layer_dose_complete',layer=l,kind='control' if control else 'neuron',dose=dose,
                    directions=width,layer_elapsed_seconds=time.monotonic()-st)
        files[name]=sha(path/name)
    save_json(path/'complete.json',dict(layer=l,neurons=e.widths[l],controls=512,doses=list(DOSES),
                                      frames=12,seconds=time.monotonic()-st,sha256=files,signature=sig,completed=now()))
    log('layer_complete',layer=l,seconds=time.monotonic()-st)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','screen']);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--layers',type=int,nargs='*');a=p.parse_args()
    precision=ensure_gate();e,c,m=load_runtime(precision)
    if a.mode=='benchmark':
        result=benchmark(e,c,m,a.batch)
        save_json(OUT/'logs'/f'benchmark_batch{a.batch}.json',result)
        log('benchmark_estimate',**{k:v for k,v in result.items() if k!='measurements' and k!='layer_estimates'})
        return
    estimate=json.loads((OUT/'logs'/f'benchmark_batch{a.batch}.json').read_text())
    assert estimate['total_estimated_hours']<=8,'Optimize or re-estimate before exhaustive screen'
    sig=signature();log('screen_start',signature=sig,batch=a.batch,estimated_hours=estimate['total_estimated_hours'])
    for l in (a.layers if a.layers is not None else range(e.L)):
        if not gpu_check(l):
            del e,c,m;gc.collect();torch.cuda.empty_cache()
            wait_free();e,c,m=load_runtime(precision)
        run_layer(e,c,m,l,a.batch,sig)
    log('screen_invocation_complete')

if __name__=='__main__':main()
