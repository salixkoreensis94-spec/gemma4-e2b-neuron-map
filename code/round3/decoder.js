/* Fixed-width data reader shared by the offline checks and static map. */
(()=>{'use strict';
 const WIDTHS=Array.from({length:35},(_,l)=>l<15?6144:12288), ROW=580;
 const base=l=>WIDTHS.slice(0,l).reduce((a,b)=>a+b,0);
 const gid=(l,n)=>base(l)+n;
 function locate(id){let l=0;while(l<35&&id>=WIDTHS[l])id-=WIDTHS[l++];if(l===35)throw Error('Invalid ID');return [l,id];}
 function decodeCard(buffer,l,n){if(l<0||l>34||n<0||n>=WIDTHS[l])throw Error('Invalid neuron');
  const v=new DataView(buffer),start=(n%256)*ROW;let p=start;
  function read(kind,count=1){const size={u32:4,u16:2,f32:4,f64:8}[kind],method={u32:'getUint32',u16:'getUint16',f32:'getFloat32',f64:'getFloat64'}[kind];let a=[];for(let i=0;i<count;i++){a.push(v[method](p,true));p+=size;}return count===1?a[0]:a;}
  const c={layer:l,neuron:n,positive:read('u32'),mean:read('f64'),maximum:read('f32'),quantiles:read('f32',6),language:read('u32',2),script:read('u32',7),sampled:read('u16'),fireIds:read('u32',10),fireCounts:read('u16',10),chunkIds:read('u32',20),offsets:read('u16',20),activations:read('f32',20),tokenIds:read('u32',20),topIds:read('u32',10),topValues:read('f32',10),bottomIds:read('u32',10),bottomValues:read('f32',10)};
  if(p-start!==578)throw Error('Schema mismatch');c.frequency=c.positive/49438284;return c;
 }
 function normalize(s){return s.normalize('NFKC').trim().toLowerCase();}
 function bucket(s){let h=2166136261;for(const b of new TextEncoder().encode(s)){h=Math.imul(h^b,16777619)>>>0;}return (h&255).toString(16).padStart(2,'0');}
 globalThis.Round3={WIDTHS,ROW,base,gid,locate,decodeCard,normalize,bucket};
})();
