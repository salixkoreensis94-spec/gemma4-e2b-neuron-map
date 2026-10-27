from config import NVIDIA_SMI
"""S2 numerical kernels with persistent, batched autoregressive KV cache."""
import math,time,json,subprocess
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader,dequantize
from s2_engine import Engine,ROOT
import kernels
OUT=ROOT/'S3b'
def gpu_guard():
    while True:
        used=int(subprocess.check_output([NVIDIA_SMI,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        own=torch.cuda.memory_reserved()/2**20 if torch.cuda.is_initialized() else 0
        external=max(0,used-own)
        with (OUT/'logs/gpu_checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),used_mib=used,own_reserved_mib=own,external_mib=external,pause=external>3072))+'\n')
        if external<=3072:return
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        time.sleep(50)

class GenerateEngine(Engine):
    def __init__(self):
        gpu_guard();super().__init__()
        self.pieces=[bytes.fromhex(x) for x in json.loads((OUT/'inputs/token_piece_hex.json').read_text())]
        self.app={}
        vector_path=OUT/'inputs/register_banmal_mean.gguf'
        if vector_path.exists():
            rd=GGUFReader(str(vector_path))
            self.app={int(t.name.split('.')[-1]):torch.from_numpy(dequantize(t.data,t.tensor_type).astype(np.float32).copy()).to('cuda').flatten() for t in rd.tensors}

    def vectors(self,arm,dose):
        result={}
        if arm['kind'] in ['app','app_control']:
            for l in range(4,15):
                v=self.app[l]
                if arm['kind']=='app_control':v=self.directions(l,[20261026%512],controls=True)[0]*torch.linalg.vector_norm(v)/self.norms[l]['median']
                result[l]=dose*v
        else:
            comp=arm['components'];k=len(comp)
            for r in comp:
                l=r['layer'];control='control_index' in r
                v=self.directions(l,[r['control_index'] if control else r['neuron']],controls=control)[0]*r['sign']*dose/math.sqrt(k)
                result[l]=result.get(l,0)+v
        return result

    def batch_inputs(self,ids):
        # PLE lookups remain on CPU exactly as S2; supports batched decode.
        ids=np.asarray(ids,dtype=np.int64);ti=torch.tensor(ids,device='cuda',dtype=torch.long)
        h=F.embedding(ti,self.embed)*float(np.sqrt(np.float32(self.D)))
        t=self.tensors['per_layer_token_embd.weight'];a=dequantize(t.data[ids.reshape(-1)],t.tensor_type).astype(np.float32,copy=False)
        pe=torch.from_numpy(a.copy()).to('cuda').reshape(*ids.shape,self.L,self.P)*math.sqrt(self.P)
        proj=self.mm(h,'per_layer_model_proj.weight')*float(np.float32(1)/np.sqrt(np.float32(self.D)))
        proj=self.norm(proj.reshape(*ids.shape,self.L,self.P),'per_layer_proj_norm.weight')
        return h,(proj+pe)*float(np.float32(1)/np.sqrt(np.float32(2)))

    def step(self,ids,pos,valid,kv,push):
        if kv:
            self.mm_fma=torch.zeros(pos.numel(),device='cuda',dtype=torch.bool)
            self.mm_sgemm=torch.zeros_like(self.mm_fma)
        else:
            lengths=valid.sum(-1)[:,None]
            chunk_lengths=(lengths-(pos//64)*64).clamp(max=64)
            self.mm_fma=((pos%64)<((chunk_lengths//4)*4)).reshape(-1)
            self.mm_sgemm=(chunk_lengths.expand_as(pos)>=4).reshape(-1)
        self.in_generation=True
        h,pe=self.batch_inputs(ids);newkv={}
        for l in range(self.L):
            p=f'blk.{l}.';s=self.swa[l];dim=self.freq[s].numel()*2
            z=self.norm(h,p+'attn_norm.weight')
            q=self.mm(z,p+'attn_q.weight').reshape(*z.shape[:-1],self.H,dim)
            q=self.rope(self.norm(q,p+'attn_q_norm.weight'),pos,s).transpose(1,2)
            if l<self.shared_start:
                k=self.mm(z,p+'attn_k.weight').reshape(*z.shape[:-1],self.KH,dim)
                v=self.mm(z,p+'attn_v.weight').reshape_as(k) if p+'attn_v.weight' in self.w else k
                k=self.rope(self.norm(k,p+'attn_k_norm.weight'),pos,s).transpose(1,2).half().float()
                v=self.norm(v).transpose(1,2).half().float()
                if kv:k=torch.cat([kv[l][0],k],2);v=torch.cat([kv[l][1],v],2)
                newkv[l]=(k,v)
            else:k,v=newkv[self.sources[s]]
            t=h.shape[1];length=k.shape[2]
            qi=torch.arange(length-t,length,device='cuda')
            ki=torch.arange(length,device='cuda')
            mask=valid[:,None,None,:] & (ki[None,:]<=qi[:,None])[None,None,:,:]
            if s:mask=mask & (ki[None,:]>qi[:,None]-512)[None,None,:,:]
            att=kernels.attention(q,k,v,mask).transpose(1,2).reshape(*h.shape[:-1],self.H*dim)
            h=h+self.norm(self.mm(att,p+'attn_output.weight'),p+'post_attention_norm.weight')
            z=self.norm(h,p+'ffn_norm.weight')
            from s2_engine import gelu
            z=gelu(self.mm(z,p+'ffn_gate.weight'))*self.mm(z,p+'ffn_up.weight')
            h=h+self.norm(self.mm(z,p+'ffn_down.weight'),p+'post_ffw_norm.weight')
            z=gelu(self.mm(h,p+'inp_gate.weight'))*pe[:,:,l]
            h=h+self.norm(self.mm(z,p+'proj.weight'),p+'post_norm.weight')
            h=h*self.w[p+'layer_output_scale.weight']
            if getattr(self,'capture_reply_mask',None) is not None and l==27:
                m=self.capture_reply_mask
                self.reply_capture=(h*m[:,:,None]).sum(1)/m.sum(1)[:,None]
                self.last_capture=h[:,-1].clone()
            if l in push:h=h+push[l]
            if getattr(self,'trace',False):self.layer_trace[l]=h[:,-1].cpu().numpy().tolist()
        return self.head(h[:,-1]),newkv

    def generate(self,prompts,push=None,max_tokens=96):
        push=push or {};start=time.monotonic();b=len(prompts);length=max(map(len,prompts))
        assert length+max_tokens<=512
        ids=np.zeros((b,length),dtype=np.int64);valid=torch.zeros((b,length),device='cuda',dtype=torch.bool)
        pos=torch.zeros((b,length),device='cuda',dtype=torch.float32)
        for i,p in enumerate(prompts):ids[i,-len(p):]=p;valid[i,-len(p):]=True;pos[i,-len(p):]=torch.arange(len(p),device='cuda')
        z,kv=self.step(ids,pos,valid,{},push)
        out=[[] for _ in prompts];active=[True]*b
        for step in range(max_tokens):
            token=z.argmax(-1).tolist()
            for i,x in enumerate(token):
                if active[i]:out[i].append(x)
                if x in (1,106):active[i]=False
            if not any(active) or step==max_tokens-1:break
            valid=torch.cat([valid,torch.ones((b,1),device='cuda',dtype=torch.bool)],1)
            pos=torch.tensor([[len(p)+step] for p in prompts],device='cuda',dtype=torch.float32)
            z,kv=self.step(np.array(token)[:,None],pos,valid,kv,push)
            if step%24==23:gpu_guard()
        replies=[]
        for tokens in out:
            raw=b''.join(self.pieces[t] for t in tokens if t not in (1,106))
            broken=0;at=0
            while at<len(raw):
                try:raw[at:].decode('utf8');break
                except UnicodeDecodeError as err:broken+=err.end-err.start;at+=err.end
            replies.append(dict(token_ids=tokens,reply=raw.decode('utf8','replace'),raw_hex=raw.hex(),broken_bytes=broken,truncated=tokens[-1] not in (1,106),length=sum(t not in (1,106) for t in tokens)))
        return replies,time.monotonic()-start
