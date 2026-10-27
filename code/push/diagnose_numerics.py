import json,os,time
import torch,numpy as np
from generate_engine import GenerateEngine,OUT,ROOT
import exact_linear
import kernels
class ExactEngine(GenerateEngine):
    def __init__(self):
        self.raw={};super().__init__()
        for name,t in self.tensors.items():
            if name=='per_layer_token_embd.weight':continue
            if int(t.tensor_type) in (12,13,14):self.raw[name]=torch.from_numpy(t.data.copy()).to('cuda')
        self.exact=True;self.fma=bool(int(os.environ.get('EXACT_FMA','0')))
        self.cpu_rope={bool(i):torch.tensor(x,device='cuda',dtype=torch.float32) for i,x in enumerate(json.loads((OUT/'inputs/rope_cpu_op.json').read_text()))} if (OUT/'inputs/rope_cpu_op.json').exists() else None
        path=OUT/'inputs/exp_cpu_corrections.bin'
        if path.exists():
            header=np.fromfile(path,dtype='<u8',count=2);n=int(header[0])
            a=np.memmap(path,dtype='<i4',mode='r',offset=16,shape=(2,n))
            kernels.CPU_EXP_KEYS=torch.from_numpy(a[0].copy()).to('cuda')
            kernels.CPU_EXP_VALUES=torch.from_numpy(a[1].copy().view('<f4')).to('cuda')
            kernels.CPU_EXP_SIZE=n
    def mm(self,x,name):
        if getattr(self,'exact',False) and int(self.tensors[name].tensor_type) in (0,12,13,14,30):
            mode=getattr(self,'mm_fma',None)
            sgem=getattr(self,'mm_sgemm',None)
            if getattr(self,'in_generation',False) and name.startswith('blk.34.ffn_'):
                mode=torch.zeros(x.numel()//x.shape[-1],device='cuda',dtype=torch.bool)
            if getattr(self,'in_generation',False) and name in ['blk.34.inp_gate.weight','blk.34.proj.weight']:
                sgem=torch.zeros(x.numel()//x.shape[-1],device='cuda',dtype=torch.bool)
            return exact_linear.linear(x,self.tensors[name],self.raw.get(name),self.w[name],self.fma,mode,sgem)
        return super().mm(x,name)
    def inputs(self,ids):
        return self.batch_inputs(np.array(ids)[None,:])
    def rope(self,x,pos,s):
        if not getattr(self,'cpu_rope',None):return super().rope(x,pos,s)
        return exact_linear.rope(x,pos,self.cpu_rope[s])
if __name__=='__main__':
    e=ExactEngine();p=json.loads((ROOT/'inputs/prompts_rendered.json').read_text())[0];start=time.monotonic()
    z,cache=e.forward(p['token_ids'],capture=True)
    out=[]
    for l,h in enumerate(cache['outputs']):
        ref=np.fromfile(ROOT/f'S2/reference/s1_0_layer_{l}.f32',dtype='<f4').reshape(h.shape)
        diff=h.cpu().numpy()-ref
        out.append(dict(layer=l,rms=float(np.sqrt(np.mean(diff**2))),max=float(abs(diff).max())))
    ref=np.fromfile(ROOT/'S2/reference/s1_0.f32',dtype='<f4')
    print('LAYERS',out,flush=True);print('LOGITS',float(abs(z.cpu().numpy()-ref).max()),'SECONDS',time.monotonic()-start,flush=True)
