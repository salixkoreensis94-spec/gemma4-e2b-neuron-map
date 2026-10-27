"""Independent float64 NumPy check of register, KL direction and entropy."""
import numpy as np

def lse(x):
    m=np.max(x)
    return float(m+np.log(np.exp(x-m).sum(dtype=np.float64)))

def check(z,baseline,frames,gpu_result):
    errors=dict(shift=0.,kl=0.,entropy_change=0.)
    for i,f in enumerate(frames):
        a=np.asarray(z[i],dtype=np.float64);b=np.asarray(baseline[i],dtype=np.float64)
        ap=a-lse(a);bp=b-lse(b);p=np.exp(ap);q=np.exp(bp)
        reg=lambda x:lse(x[f['polite_ids']])-lse(x[f['banmal_ids']])
        ref=dict(shift=reg(a)-reg(b),kl=float(np.sum(p*(ap-bp),dtype=np.float64)),
                 entropy_change=float(-np.sum(p*ap,dtype=np.float64)+np.sum(q*bp,dtype=np.float64)))
        for kind,key in [('shift','shifts'),('kl','kl'),('entropy_change','entropy_change')]:
            errors[kind]=max(errors[kind],abs(ref[kind]-float(gpu_result[key][0,i])))
    assert max(errors.values())<2e-4,errors
    return errors
