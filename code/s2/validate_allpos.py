import csv
import gzip
import json
import time
import gc
import numpy as np
import torch
from engine import Engine,OUT,ROOT,save_json
from metrics import Metrics
from run_screen import ensure_gate,gpu_check,wait_free,log,load_runtime,sha
from independent_metrics import check as independent_metric_check

def ranks(a):
    order=np.argsort(a,kind='stable');r=np.empty(len(a),dtype=float)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2;i=j
    return r

def correlate(rows):
    x=np.array([r['last_only'] for r in rows]).reshape(-1)
    y=np.array([r['all_positions'] for r in rows]).reshape(-1)
    return dict(observations=len(x),pearson=float(np.corrcoef(x,y)[0,1]),
                spearman=float(np.corrcoef(ranks(x),ranks(y))[0,1]),
                sign_agreement=float(np.mean(np.sign(x)==np.sign(y))),
                mean_absolute_effect_difference=float(np.mean(np.abs(x-y))))

def main():
    start=time.monotonic();precision=ensure_gate()
    selection=json.loads((OUT/'validation_selection.json').read_text())
    frames=json.loads((OUT/'frames.json').read_text())
    engine,cache,metrics=load_runtime(precision)
    baseline=cache['base'].cpu().numpy().astype('<f4')
    baseline_path=OUT/'baseline_logits.f32'
    if baseline_path.exists():
        assert np.array_equal(np.fromfile(baseline_path,dtype='<f4').reshape(baseline.shape),baseline)
    else:
        with baseline_path.open('xb') as f:f.write(baseline.tobytes())
    if not (OUT/'baseline_stats.json').exists():
        stats=[]
        for i,f in enumerate(frames):
            top=int(cache['base'][i].argmax().item())
            stats.append(dict(id=f['id'],name=f['name'],register_statistic=float(metrics.baseR[i].item()),
                              entropy=float(metrics.ent[i].item()),top1_id=top,top1_surface=metrics.tokens[top],
                              top1_korean_proxy=bool(metrics.plausible[top]),
                              polite_ending_probability=float(metrics.lp[i,f['polite_ids']].exp().sum().item()),
                              banmal_ending_probability=float(metrics.lp[i,f['banmal_ids']].exp().sum().item())))
        save_json(OUT/'baseline_stats.json',dict(shape=list(baseline.shape),dtype='little-endian float32',
                                                engine_validation_sha256=sha(OUT/'engine_validation.json'),frames=stats))
    checkpoint_dir=OUT/'allpos_layers';checkpoint_dir.mkdir(exist_ok=True)
    signature={name:sha(OUT/name) for name in ('validation_selection.json','engine_validation.json','frames.json')}
    groups={}
    for row in selection['neurons']:groups.setdefault(row['layer'],[]).append(row)
    rows=[];metric_checks=[];replay_checks=[]
    for l,neurons in sorted(groups.items()):
        checkpoint=checkpoint_dir/f'layer_{l:02d}.json'
        if checkpoint.exists():
            previous=json.loads(checkpoint.read_text());assert previous['signature']==signature
            rows.extend(previous['rows']);metric_checks.append(previous['metric_check']);replay_checks.append(previous['replay_check']);log('allpos_skip_complete',layer=l);continue
        if not gpu_check(l):
            del engine,cache,metrics
            gc.collect();torch.cuda.empty_cache()
            wait_free();engine,cache,metrics=load_runtime(precision)
        # Normalize the same full strided matrix as screening, THEN select.
        # Reducing a contiguous subset can change f32 rounding and hence Q8
        # activation boundaries; replay must use exactly the screened vectors.
        directions=engine.directions(l)[[r['neuron'] for r in neurons]]
        observations={}
        metric_check=None
        for d in (-.8,-.5,.5,.8):
            hidden=engine.tail(cache,l,directions*d)
            last_result=metrics.evaluate_hidden(hidden);last=last_result['shifts']
            if d==-.5:
                sample_logits=engine.head(hidden[:12]).cpu().numpy()
                metric_check=dict(layer=l,neuron=neurons[0]['neuron'],dose=d,
                                  max_abs_errors=independent_metric_check(sample_logits,baseline,frames,last_result))
                metric_checks.append(metric_check)
            for j,r in enumerate(neurons):
                observations[(j,d)]=dict(layer=l,neuron=r['neuron'],subset=r['subset'],dose=d,
                                          last_only=last[j].tolist(),all_positions=[None]*12,allpos_top1=[None]*12)
        for fi,f in enumerate(frames):
            _,raw=engine.forward(f['token_ids'],capture=True)
            for d in (-.8,-.5,.5,.8):
                # At most eight perturbations per all-position attention batch.
                for at in range(0,len(neurons),8):
                    h=engine.full_tail(raw,l,directions[at:at+8]*d)
                    z=engine.head(h)
                    ff=torch.full((len(h),),fi,device='cuda',dtype=torch.long)
                    delta=(metrics.register(z,ff)-metrics.baseR[fi]).cpu().numpy()
                    top=z.argmax(-1).cpu().tolist()
                    for j in range(len(h)):
                        observations[(at+j,d)]['all_positions'][fi]=float(delta[j])
                        observations[(at+j,d)]['allpos_top1'][fi]=top[j]
            del raw
        layer_rows=list(observations.values());rows.extend(layer_rows)
        max_replay=0.;replayed=0
        lookup={(r['neuron'],r['dose']):r for r in layer_rows}
        with gzip.open(OUT/'layers'/f'layer_{l:02d}'/'screen.tsv.gz','rt') as f:
            for saved in csv.DictReader(f,delimiter='\t'):
                key=(int(saved['neuron']),float(saved['dose']))
                if key not in lookup:continue
                effect=np.array(json.loads(saved['shifts']))
                error=float(np.max(np.abs(effect-np.array(lookup[key]['last_only']))))
                max_replay=max(max_replay,error);replayed+=1
        assert replayed==len(layer_rows) and max_replay<1e-5,(l,replayed,max_replay)
        replay_check=dict(layer=l,rows=replayed,max_abs_shift_diff=max_replay);replay_checks.append(replay_check)
        save_json(checkpoint,dict(signature=signature,layer=l,rows=layer_rows,metric_check=metric_check,replay_check=replay_check))
        log('allpos_layer_complete',layer=l,neurons=len(neurons),seconds=time.monotonic()-start)
    correlations={}
    for subset in ('strongest','random','all','early_layers_0_to_13','shared_kv_layers_14_to_34'):
        rr=[r for r in rows if subset=='all' or r['subset']==subset or
            (subset=='early_layers_0_to_13' and r['layer']<14) or
            (subset=='shared_kv_layers_14_to_34' and r['layer']>=14)]
        if not rr:continue
        correlations[subset]=dict(pooled=correlate(rr),by_dose={str(d):correlate([r for r in rr if r['dose']==d]) for d in (-.8,-.5,.5,.8)})
    save_json(OUT/'validation_allpos.json',dict(selection=selection,rows=rows,correlations=correlations,
                                              independent_float64_metric_checks=metric_checks,screen_replay_checks=replay_checks,
                                              direction_convention='normalize full strided W_down.T exactly as screening, then select rows',
                                              seconds=time.monotonic()-start,scope='same l_out addition at every prompt position'))
    log('allpos_complete',seconds=time.monotonic()-start,correlations=correlations)

if __name__=='__main__':main()
