"""Minimal text-only Gemma4 ISWA forward, from the S1 GGUF and graph.

No from_pretrained, Hub access, re-quantization, learned config or HF weights.
GGUF norms already contain the Gemma +1 offset. Keep them unchanged.
The large packed PLE embedding stays mmap'ed on CPU; select prompt rows only.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader, dequantize
import kernels

from config import ROOT, MODEL
OUT=ROOT/'S2'


def save_json(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def rms(x,w=None,eps=1e-6):return kernels.rms(x,w,eps)

GELU_TABLE=None
def gelu(x):return kernels.gelu(x,GELU_TABLE)

def q8k(x):
    return kernels.q8k(x)

class Engine:
    def __init__(self,precision='highest'):
        global GELU_TABLE
        torch.set_grad_enabled(False)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision(precision)
        self.precision=precision
        GELU_TABLE=torch.from_numpy(np.fromfile(ROOT/'inputs/gelu_table.f16',dtype='<f2')).to('cuda')
        self.reader=GGUFReader(str(MODEL),'r')
        self.tensors={t.name:t for t in self.reader.tensors}
        self.meta={k:f.contents() for k,f in self.reader.fields.items() if not k.startswith('tokenizer.')}
        self.D=int(self.meta['gemma4.embedding_length'])
        self.L=int(self.meta['gemma4.block_count'])
        self.H=int(self.meta['gemma4.attention.head_count'])
        self.KH=int(self.meta['gemma4.attention.head_count_kv'])
        self.shared_start=self.L-int(self.meta['gemma4.attention.shared_kv_layers'])
        self.swa=list(self.meta['gemma4.attention.sliding_window_pattern'])
        self.eps=float(self.meta['gemma4.attention.layer_norm_rms_epsilon'])
        self.cap=float(self.meta['gemma4.final_logit_softcapping'])
        self.P=int(self.meta['gemma4.embedding_length_per_layer_input'])
        self.norms=json.loads((ROOT/'inputs/residual_norms.json').read_text())['layers']
        self.widths=[int(self.tensors[f'blk.{l}.ffn_down.weight'].shape[0]) for l in range(self.L)]
        self.sources={s:max(i for i in range(self.shared_start) if self.swa[i]==s) for s in (False,True)}
        self.w={}
        begin=time.monotonic()
        for name,t in self.tensors.items():
            if name=='per_layer_token_embd.weight':continue
            a=dequantize(t.data,t.tensor_type).astype(np.float32,copy=False)
            self.w[name]=torch.from_numpy(a.copy()).to('cuda')
            del a
            if name.endswith('ffn_down.weight'):
                print('loaded',name,round(torch.cuda.memory_allocated()/2**30,3),'GiB',flush=True)
        self.embed=self.w['token_embd.weight']
        self.freq={}
        for s in (True,False):
            d=int(self.meta['gemma4.attention.key_length_swa' if s else 'gemma4.attention.key_length'])
            base=float(self.meta['gemma4.rope.freq_base_swa' if s else 'gemma4.rope.freq_base'])
            inv=base**(-torch.arange(0,d,2,device='cuda',dtype=torch.float32)/d)
            if not s:inv=inv/self.w['rope_freqs.weight'].flatten()
            self.freq[s]=inv
        gc.collect()
        print('engine loaded',time.monotonic()-begin,'seconds; allocated',torch.cuda.memory_allocated()/2**30,flush=True)

    def norm(self,x,name=None):return rms(x,None if name is None else self.w[name],self.eps)
    def mm(self,x,name):
        ty=int(self.tensors[name].tensor_type)
        if ty in (12,13,14):x=q8k(x)
        elif ty==30:x=x.to(torch.bfloat16).float()
        return kernels.linear(x,self.w[name])
    def rope(self,x,pos,s):
        theta=pos[...,None]*self.freq[s]
        cs=theta.cos().unsqueeze(-2)
        sn=theta.sin().unsqueeze(-2)
        a,b=x.chunk(2,-1)
        return torch.cat((a*cs-b*sn,b*cs+a*sn),-1)

    def inputs(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        ti=torch.tensor(ids,device='cuda',dtype=torch.long)
        h=F.embedding(ti,self.embed).unsqueeze(0)*math.sqrt(self.D)
        t=self.tensors['per_layer_token_embd.weight']
        a=dequantize(t.data[ids],t.tensor_type).astype(np.float32,copy=False)
        pe=torch.from_numpy(a.copy()).to('cuda').reshape(1,len(ids),self.L,self.P)*math.sqrt(self.P)
        proj=self.mm(h,'per_layer_model_proj.weight')/math.sqrt(self.D)
        proj=self.norm(proj.reshape(1,len(ids),self.L,self.P),'per_layer_proj_norm.weight')
        pe=(proj+pe)/math.sqrt(2)
        return h,pe

    def block(self,h,pe,l,pos,kv,cache=None):
        p=f'blk.{l}.'
        s=self.swa[l]
        dim=self.freq[s].numel()*2
        z=self.norm(h,p+'attn_norm.weight')
        q=self.mm(z,p+'attn_q.weight').reshape(*z.shape[:-1],self.H,dim)
        q=self.rope(self.norm(q,p+'attn_q_norm.weight'),pos,s).transpose(1,2)
        if l<self.shared_start:
            k=self.mm(z,p+'attn_k.weight').reshape(*z.shape[:-1],self.KH,dim)
            v=self.mm(z,p+'attn_v.weight').reshape_as(k) if p+'attn_v.weight' in self.w else k
            k=self.rope(self.norm(k,p+'attn_k_norm.weight'),pos,s).transpose(1,2)
            v=self.norm(v).transpose(1,2)
            k=k.half().float();v=v.half().float()
            if cache is not None:
                # Prefix cache ends immediately BEFORE this final prompt token.
                ck,cv=cache['kv'][l]
                count=h.shape[0]//cache['frames']
                k=torch.cat((ck[:,:,:-1].repeat(count,1,1,1),k),dim=2)
                v=torch.cat((cv[:,:,:-1].repeat(count,1,1,1),v),dim=2)
            kv[l]=(k,v)
        else:
            source=self.sources[s]
            if source in kv:k,v=kv[source]
            else:
                ck,cv=cache['kv'][source]
                count=h.shape[0]//cache['frames']
                k=ck.repeat(count,1,1,1);v=cv.repeat(count,1,1,1)
        # All frames are <512 tokens, so causal attention covers the SWA window.
        if cache is None:
            length=h.shape[1]
            mask=torch.ones(1,1,length,length,device='cuda',dtype=torch.bool).tril()
        else:
            count=h.shape[0]//cache['frames']
            mask=cache['mask'].repeat(count,1,1,1)
        att=kernels.attention(q,k,v,mask)
        att=att.transpose(1,2).reshape(*h.shape[:-1],self.H*dim)
        h=h+self.norm(self.mm(att,p+'attn_output.weight'),p+'post_attention_norm.weight')
        z=self.norm(h,p+'ffn_norm.weight')
        z=gelu(self.mm(z,p+'ffn_gate.weight'))*self.mm(z,p+'ffn_up.weight')
        h=h+self.norm(self.mm(z,p+'ffn_down.weight'),p+'post_ffw_norm.weight')
        z=gelu(self.mm(h,p+'inp_gate.weight'))*pe
        h=h+self.norm(self.mm(z,p+'proj.weight'),p+'post_norm.weight')
        return h*self.w[p+'layer_output_scale.weight']

    def head(self,h):
        z=self.mm(self.norm(h,'output_norm.weight'),'token_embd.weight')
        return self.cap*torch.tanh(z/self.cap)

    def forward(self,ids,push_layer=None,push=None,scope='all',capture=False):
        h,pe=self.inputs(ids)
        pos=torch.arange(len(ids),device='cuda',dtype=torch.float32)[None,:]
        kv={};outputs=[]
        for l in range(self.L):
            h=self.block(h,pe[:,:,l],l,pos,kv)
            if l==push_layer:
                if scope=='last':h[:,-1]+=push
                else:h=h+push
            if capture:outputs.append(h.clone())
        return self.head(h[:,-1]),dict(outputs=outputs,kv=kv,pe=pe,pos=pos)

    def make_cache(self,frames):
        raw=[];baselines=[]
        for f in frames:
            z,c=self.forward(f['token_ids'],capture=True)
            baselines.append(z)
            raw.append(c)
        count=len(frames);length=max(len(f['token_ids']) for f in frames)
        assert length<=512
        kv={}
        for l in range(self.shared_start):
            ks=[];vs=[]
            for c in raw:
                k,v=c['kv'][l];pad=length-k.shape[2]
                ks.append(F.pad(k,(0,0,pad,0)))
                vs.append(F.pad(v,(0,0,pad,0)))
            kv[l]=(torch.cat(ks),torch.cat(vs))
        mask=torch.zeros(count,1,1,length,device='cuda',dtype=torch.bool)
        for i,f in enumerate(frames):mask[i,:,:,-len(f['token_ids']):]=True
        cache=dict(frames=count,kv=kv,mask=mask,
                   outputs=[torch.cat([c['outputs'][l][:,-1:] for c in raw]) for l in range(self.L)],
                   pe=torch.cat([c['pe'][:,-1:] for c in raw]),
                   pos=torch.tensor([[len(f['token_ids'])-1] for f in frames],device='cuda',dtype=torch.float32),
                   base=torch.cat(baselines))
        del raw
        return cache

    def tail(self,cache,l,push):
        # push shape [directions, D], ordering direction-major then frame.
        count=push.shape[0];frames=cache['frames']
        h=(cache['outputs'][l][None,:,:,:]+push[:,None,None,:]).reshape(count*frames,1,self.D)
        pe=cache['pe'].repeat(count,1,1,1)
        pos=cache['pos'].repeat(count,1)
        kv={}
        for j in range(l+1,self.L):h=self.block(h,pe[:,:,j],j,pos,kv,cache)
        return h[:,0]

    def full_tail(self,raw,l,push):
        h=raw['outputs'][l]+push[:,None,:]
        kv=dict(raw['kv'])
        for j in range(l+1,self.L):h=self.block(h,raw['pe'][:,:,j],j,raw['pos'],kv)
        return h[:,-1]

    def directions(self,l,indices=None,controls=False):
        if controls:
            a=np.memmap(ROOT/'inputs/control_directions.f32',dtype='<f4',mode='r',shape=(35,512,1536))[l]
            x=torch.tensor(np.array(a if indices is None else a[indices]),device='cuda')
        else:
            x=self.w[f'blk.{l}.ffn_down.weight'].T
            if indices is not None:x=x[indices]
        return x/torch.linalg.vector_norm(x,dim=1,keepdim=True)*self.norms[l]['median']

if __name__=='__main__':
    engine=Engine()
    frames=json.loads((OUT/'frames.json').read_text())
    z,c=engine.forward(frames[0]['token_ids'],capture=True)
    print('top10',torch.topk(z,10).indices.tolist(),flush=True)
