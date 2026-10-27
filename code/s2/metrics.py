import re
import numpy as np
import torch
import torch.nn.functional as F

class Metrics:
    def __init__(self,engine,frames,cache):
        self.engine=engine;self.frames=frames;self.F=len(frames)
        tok=engine.reader.get_field('tokenizer.ggml.tokens').contents()
        types=engine.reader.get_field('tokenizer.ggml.token_type').contents()
        self.tokens=[t.decode('utf-8') if isinstance(t,bytes) else t for t in tok]
        self.plausible=[]
        for t,ty in zip(self.tokens,types):
            s=t.replace('▁',' ')
            self.plausible.append(bool(ty==1 and re.search('[가-힣]',s) and
                                       re.fullmatch('[가-힣ᄀ-ᇿ㄰-㆏\\s.!?~…]+',s)))
        self.plausible=np.asarray(self.plausible,dtype=bool)
        self.lp=F.log_softmax(cache['base'],dim=-1)
        self.ent=-(self.lp.exp()*self.lp).sum(-1)
        self.ids={};self.masks={}
        for kind in ('banmal','polite'):
            width=max(len(f[kind+'_ids']) for f in frames)
            ids=[];masks=[]
            for f in frames:
                a=f[kind+'_ids'];ids.append(a+[0]*(width-len(a)))
                masks.append([True]*len(a)+[False]*(width-len(a)))
            self.ids[kind]=torch.tensor(ids,device='cuda')
            self.masks[kind]=torch.tensor(masks,device='cuda',dtype=torch.bool)
        self.baseR=self.register(cache['base'],torch.arange(self.F,device='cuda'))

    def register(self,z,fi):
        terms=[]
        for kind in ('polite','banmal'):
            x=z.gather(1,self.ids[kind][fi]).masked_fill(~self.masks[kind][fi],-torch.inf)
            terms.append(torch.logsumexp(x,dim=1))
        return terms[0]-terms[1]

    def evaluate_hidden_gpu(self,h,chunk=96):
        n=h.shape[0]
        result={k:[] for k in ('shifts','kl','entropy_change','top1')}
        for start in range(0,n,chunk):
            z=self.engine.head(h[start:start+chunk])
            fi=torch.arange(start,start+z.shape[0],device='cuda')%self.F
            lp=F.log_softmax(z,dim=-1)
            prob=lp.exp()
            kl=(prob*(lp-self.lp[fi])).sum(-1)
            ent=-(prob*lp).sum(-1)-self.ent[fi]
            shift=self.register(z,fi)-self.baseR[fi]
            top=z.argmax(-1)
            for k,x in zip(result,(shift,kl,ent,top)):
                result[k].append(x)
        return {k:torch.cat(v).reshape(-1,self.F) for k,v in result.items()}

    def to_cpu(self,result):
        result={k:v.cpu().numpy().copy() for k,v in result.items()}
        result['plausible']=self.plausible[result['top1']]
        for k in ('shifts','kl','entropy_change'):
            if not np.isfinite(result[k]).all():raise RuntimeError('nonfinite '+k)
        if result['kl'].min() < -1e-4:raise RuntimeError('negative KL')
        return result

    def evaluate_hidden(self,h,chunk=96):
        return self.to_cpu(self.evaluate_hidden_gpu(h,chunk))

class GraphEvaluator:
    def __init__(self,engine,cache,metrics,layer,batch):
        self.metrics=metrics
        self.push=torch.zeros(batch,engine.D,device='cuda')
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                metrics.evaluate_hidden_gpu(engine.tail(cache,layer,self.push))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=metrics.evaluate_hidden_gpu(engine.tail(cache,layer,self.push))
        # Verify replay against eager computation on a nonzero direction.
        test=engine.directions(layer,list(range(batch)))*.5
        ref=metrics.evaluate_hidden(engine.tail(cache,layer,test))
        got=self(test)
        for k in ('shifts','kl','entropy_change'):
            err=float(np.abs(ref[k]-got[k]).max())
            if err>1e-5:raise RuntimeError('CUDA graph parity '+k+': '+str(err))
        if not np.array_equal(ref['top1'],got['top1']):raise RuntimeError('CUDA graph top1 parity')
        # Release warm-up allocator blocks; the graph's private pool stays live.
        torch.cuda.empty_cache()

    def __call__(self,push):
        assert push.shape==self.push.shape
        self.push.copy_(push)
        self.graph.replay()
        return self.metrics.to_cpu(self.output)
