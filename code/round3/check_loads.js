/* Stub fetch reads local data only. Output is counts/IDs, never corpus text. */
'use strict';
const {build}=require('./check_page.js'),assert=require('assert');
const button=element=>element.all(x=>x.tagName==='BUTTON')[0];
const pathOf=(type,name)=>`data/round3/stats/${type}/${name}`;
function fault(target,mode){let enabled=true,chosen=null;const counts=new Map(),delays=[];return {
  counts,delays,get chosen(){return chosen;},serve(){enabled=false;},
  setTimeout(fn,ms){delays.push(ms);return setTimeout(fn,0);},
  async fetch(name){counts.set(name,(counts.get(name)||0)+1);if(!target(name))return;
    if(chosen===null)chosen=name;if(name!==chosen||!enabled)return;
    const attempt=counts.get(name);
    if(mode==='once-network'&&attempt===1)throw TypeError('Synthetic network failure');
    if(mode==='once-503'&&attempt===1)return new Response('',{status:503});
    if(mode==='network')throw TypeError('Synthetic network failure');
    if(mode==='404')return new Response('',{status:404});
    if(mode==='corrupt-gzip')return new Response(new Uint8Array([0,1,2]));
    if(mode==='invalid-json')return new Response('{');
  }
};}
function options(stub,extra={}){return {fetch:stub.fetch,setTimeout:stub.setTimeout,...extra};}
function rows(app,layer){const {elements:e,context:c}=app;assert.equal(c.Round3Browse.getState().layer,layer);assert.equal(e['list-body'].children.length,50);assert(e['list-body'].children.every(row=>Number(row.dataset.layer)===layer));assert(e['list-error'].hidden);}
function shown(app,kind,lang='ko'){const element=app.elements[kind==='card'?'card':kind+'-error'];assert(!element.hidden);assert(element.textContent.includes(app.context.Round3Reader.strings[lang].loadFailed));assert.equal(button(element).textContent,app.context.Round3Reader.strings[lang].tryAgain);return button(element);}
async function main(){
  let recovery=0,manual=0,viewFailures=0,races=0;const matrix=[];
  for(const lang of ['ko','en'])for(const viewport of [1024,375]){
    for(const mode of ['once-network','once-503']){
      const stub=fault(name=>name.includes('/vocab/'),mode),app=await build(undefined,options(stub,{viewport}));
      await app.elements.locale.onchange({target:{value:lang}});await app.context.Round3View.navigate('#L5-list');rows(app,5);
      assert.equal(stub.counts.get(stub.chosen),2);assert.deepEqual(stub.delays,[250]);
      await app.context.Round3View.navigate('#L5-list');assert.equal(stub.counts.get(stub.chosen),2);recovery++;
    }
    // A rejected shared vocabulary promise is evicted even after both attempts fail.
    const stub=fault(name=>name.includes('/vocab/'),'network'),app=await build(undefined,options(stub,{viewport}));
    await app.elements.locale.onchange({target:{value:lang}});await app.context.Round3View.navigate('#L5-list');
    const retry=shown(app,'list',lang);assert.equal(stub.counts.get(stub.chosen),2);assert.equal(app.elements['list-body'].children.length,0);
    stub.serve();await retry.onclick();rows(app,5);assert.equal(stub.counts.get(stub.chosen),3);manual++;
    // 404 has no automatic delay/retry. A different layer still works.
    const missing=pathOf('browse','L05.bin.gz'),notFound=fault(name=>name===missing,'404'),other=await build(undefined,options(notFound,{viewport}));
    await other.elements.locale.onchange({target:{value:lang}});await other.context.Round3View.navigate('#L5-list');const oldRetry=shown(other,'list',lang);
    assert.equal(notFound.counts.get(missing),1);assert.equal(notFound.delays.length,0);
    await other.context.Round3View.navigate('#L15-list');rows(other,15);await oldRetry.onclick();rows(other,15);
    notFound.serve();await other.context.Round3View.navigate('#L5-list');rows(other,5);assert.equal(notFound.counts.get(missing),2);
    matrix.push({language:lang,viewport});
  }
  // Even a shared vocab 404 must not poison the next layer after service recovers.
  const shared=fault(name=>name.includes('/vocab/'),'404'),next=await build(undefined,options(shared));
  await next.context.Round3View.navigate('#L5-list');shown(next,'list');assert.equal(shared.counts.get(shared.chosen),1);assert.equal(shared.delays.length,0);
  shared.serve();await next.context.Round3View.navigate('#L15-list');rows(next,15);await next.context.Round3View.navigate('#L5-list');rows(next,5);assert.equal(shared.counts.get(shared.chosen),2);
  // Every requested view has local recovery, including the two browse promise wrappers.
  for(const [kind,route,target] of [
    ['whole','#',name=>name.endsWith('/browse/overview.bin.gz')],
    ['whole','#',name=>name.endsWith('/browse/metadata.json')],
    ['grid','#L5',name=>name.endsWith('/catalog/L05.bin.gz')],
    ['card','#L5-N0',name=>name.endsWith('/cards/L05_00.bin.gz')],
  ]){
    const stub=fault(target,'404'),app=await build(undefined,options(stub));
    if(kind!=='whole')await app.context.Round3View.navigate(route);
    const retry=shown(app,kind);assert.equal(stub.counts.get(stub.chosen),1);assert.equal(stub.delays.length,0);
    stub.serve();await retry.onclick();assert.equal(stub.counts.get(stub.chosen),2);
    if(kind==='card')assert.equal(app.elements.card.children[0].textContent,'L5 · 뉴런 0');
    else{assert(app.elements[kind+'-error'].hidden);assert(!app.elements[kind==='grid'?'grid-box':'whole-box'].hidden);}
    viewFailures++;
  }
  // Decode failures also evict cache entries; they are not transient fetch retries.
  for(const [mode,target,route,kind] of [
    ['corrupt-gzip',name=>name.includes('/vocab/'),'#L5-list','list'],
    ['invalid-json',name=>name.endsWith('/browse/metadata.json'),'#','whole'],
    ['invalid-json',name=>name.endsWith('/schema.json'),'#','whole'],
  ]){
    const stub=fault(target,mode),app=await build(undefined,options(stub));
    if(kind==='list')await app.context.Round3View.navigate(route);
    const retry=shown(app,kind);assert.equal(stub.counts.get(stub.chosen),1);assert.equal(stub.delays.length,0);
    stub.serve();await retry.onclick();assert.equal(stub.counts.get(stub.chosen),2);
    if(kind==='list')rows(app,5);else assert(app.elements['whole-error'].hidden);
    manual++;
  }
  // A delayed rejected old layer cannot overwrite a newer layer's list or status.
  for(const eventual of ['404','503','success']){
    let release,entered;const started=new Promise(resolve=>entered=resolve),delayed=new Promise(resolve=>release=resolve);let count=0;
    const target=pathOf('catalog','L05.bin.gz');
    const app=await build(undefined,{fetch:async name=>{if(name!==target)return;count++;if(count===1){entered();await delayed;}if(eventual==='404')return new Response('',{status:404});if(eventual==='503')return new Response('',{status:503});},setTimeout:(fn,ms)=>setTimeout(fn,0)});
    const old=app.context.Round3View.navigate('#L5-list');await started;
    await app.context.Round3View.navigate('#L15-list');await app.context.Round3View.navigate('#L27-list');rows(app,27);
    release();await old;rows(app,27);assert.equal(app.location.hash,'#L27-list');assert.equal(app.elements.status.textContent,'L27');assert(!app.elements.card.scrolled);
    assert.equal(count,eventual==='503'?2:1);races++;
  }
  // A pending card fetch and an already-created retry button cannot resurrect a card.
  let releaseCard,enteredCard;const cardStarted=new Promise(resolve=>enteredCard=resolve),cardDelay=new Promise(resolve=>releaseCard=resolve);
  const card=await build(undefined,{fetch:async name=>{if(!name.endsWith('/cards/L05_00.bin.gz'))return;enteredCard();await cardDelay;return new Response('',{status:404});}});
  const oldCard=card.context.Round3View.navigate('#L5-N0');await cardStarted;await card.context.Round3View.navigate('#L27-list');releaseCard();await oldCard;rows(card,27);assert(card.elements.card.hidden);races++;
  const stale=fault(name=>name.endsWith('/cards/L05_00.bin.gz'),'404'),saved=await build(undefined,options(stale));await saved.context.Round3View.navigate('#L5-N0');const staleRetry=shown(saved,'card');await saved.context.Round3View.navigate('#L27-list');stale.serve();await staleRetry.onclick();rows(saved,27);assert(saved.elements.card.hidden);races++;
  console.log(JSON.stringify({passed:true,matrix,automatic_recoveries:recovery,manual_recoveries:manual,in_view_failure_cases:viewFailures,race_cases:races,retry_delay_ms:250,maximum_automatic_retries:1,no_404_retry:true,shared_vocab_404_next_layer_recovers:true,success_cache_retained:true,rejected_json_and_gzip_evicted:true,browse_promises_recover:true,viewport_measurement:'existing JS/DOM model; not a browser measurement',context_text_printed:false}));
}
main().catch(error=>{console.error(JSON.stringify({passed:false,frames:String(error.stack).split('\n').filter(s=>s.trim().startsWith('at ')).map(s=>s.match(/:(\d+):(\d+)/)?.[0]).filter(Boolean),context_text_printed:false}));process.exitCode=1;});
