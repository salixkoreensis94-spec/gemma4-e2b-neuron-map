/* Local JS/DOM checks; counts and IDs only, no corpus or snippet output. */
'use strict';
const {build}=require('./check_page.js'),fs=require('fs'),path=require('path'),zlib=require('zlib'),assert=require('assert');
const root=path.resolve(__dirname,'../..'),pages=path.join(root,'docs/map'),fields=['token','layer','neuron','language','script','fmin','fmax','mmin','mmax'];
const unzip=name=>zlib.gunzipSync(fs.readFileSync(path.join(pages,'data/round3/stats',name)));
const countFrom=e=>Number(e['list-page'].textContent.match(/[\d,]+(?=개 일치| matches)/)[0].replaceAll(',',''));
const button=element=>element.all(x=>x.tagName==='BUTTON')[0];
async function main(){
  const html=fs.readFileSync(path.join(pages,'index.html'),'utf8'),css=fs.readFileSync(path.join(pages,'app.css'),'utf8');
  assert(/<form[^>]+id="search-form"[^>]+autocomplete="off"/.test(html));
  assert(html.indexOf('id="active-filters"')<html.indexOf('id="grid-view"'));
  assert(html.indexOf('id="active-filters"')<html.indexOf('id="list-view"'));
  assert(css.includes('.active-filters{display:flex;flex-wrap:wrap;'));
  assert(css.includes('.active-filters span{min-width:0;overflow-wrap:anywhere}'));
  let cases=0,restores=0;const totals=[];
  for(const lang of ['ko','en'])for(const viewport of [1024,375]){
    const app=await build(undefined,{hash:'#L5-list',viewport}),{context:c,elements:e}=app;
    await e.locale.onchange({target:{value:lang}});
    const Q=c.Round3Reader.strings[lang];assert.equal(e['label-layer'].textContent,Q.searchLayer);
    // Old failure: the search layer cannot remove browse rows, in either view.
    e.layer.value='27';await e.layer.onchange();
    for(const layer of [5,15,27,34]){
      await c.Round3View.navigate(`#L${layer}-list`);assert.equal(e['list-body'].children.length,50);
      assert(e['list-body'].children.every(row=>Number(row.dataset.layer)===layer));
      assert.equal(countFrom(e),c.Round3.WIDTHS[layer]);assert(e['active-filters'].hidden);
      await c.Round3View.navigate(`#L${layer}`);assert(e['grid-count'].textContent.includes(': '+c.Round3.WIDTHS[layer].toLocaleString()+' '));
      assert.equal(e['browse-layer'].value,String(layer));
    }
    // Find still honors the search layer.
    await e['search-form'].onsubmit({preventDefault(){}});
    assert.equal(app.location.hash,'#search');assert.equal(countFrom(e),c.Round3.WIDTHS[27]);
    assert(e['list-body'].children.every(row=>Number(row.dataset.layer)===27));
    assert(e['active-filters'].textContent.includes(Q.searchLayer+' L27'));
    // Compare the word filter with the index independently; never print token text.
    await c.Round3View.navigate('#L23-list');e.token.value='how';await e.token.oninput();
    const ids=new Set(JSON.parse(unzip(`index/${c.Round3.bucket('how')}.json.gz`))['how']||[]);
    const expected=Array.from({length:c.Round3.WIDTHS[23]},(_,n)=>n).filter(n=>ids.has(c.Round3.gid(23,n)));
    assert(expected.length>0&&expected.length<c.Round3.WIDTHS[23]);assert.equal(countFrom(e),expected.length);
    assert.deepEqual(e['list-body'].children.map(row=>Number(row.dataset.neuron)),expected.slice(0,50));
    assert(!e['active-filters'].hidden);assert(e['active-filters'].textContent.includes('how'));
    assert(e['active-filters'].textContent.startsWith(lang==='ko'?'필터 적용 중:':'Active filters:'));
    assert.equal(button(e['active-filters']).textContent,Q.clearFilters);
    await c.Round3View.navigate('#L23');assert(!e['active-filters'].hidden);assert(e['active-filters'].scrolled);
    assert(e['grid-count'].textContent.includes(': '+expected.length.toLocaleString()+' '));
    await button(e['active-filters']).onclick();assert(fields.every(id=>e[id].value===''));
    assert(e['active-filters'].hidden);assert(e['grid-count'].textContent.includes(': '+c.Round3.WIDTHS[23].toLocaleString()+' '));
    // Clearing restores both layer widths and the 50-row page.
    for(const layer of [0,23]){await c.Round3View.navigate(`#L${layer}-list`);assert.equal(countFrom(e),c.Round3.WIDTHS[layer]);assert.equal(e['list-body'].children.length,50);}
    // Every effective condition, including the neuron field, is visible.
    e.token.value='how';e.neuron.value='0';e.language.value='ko';e.script.value='Hangul';e.fmin.value='0';e.fmax.value='0.01';e.mmin.value='1000000';e.mmax.value='1000001';
    await c.Round3View.filtersChanged();assert.equal(countFrom(e),0);assert(!e['list-empty'].hidden);
    for(const term of ['how','ko',Q.neuron,Q.script,Q.fmin,Q.fmax,Q.mmin,Q.mmax]){
      assert(e['active-filters'].textContent.includes(term));assert(e['list-empty'].textContent.includes(term));
    }
    assert.equal(button(e['list-empty']).textContent,Q.clearFilters);await button(e['list-empty']).onclick();
    assert(e['list-empty'].hidden);assert(e['active-filters'].hidden);assert(fields.every(id=>e[id].value===''));
    // Clearing search also recomputes results, so old search rows cannot survive.
    e.layer.value='27';e.neuron.value='1939';await e['search-form'].onsubmit({preventDefault(){}});assert.equal(countFrom(e),1);
    await button(e['active-filters']).onclick();assert.equal(countFrom(e),337920);
    // Sort, tap and card regressions at each language/width combination.
    await c.Round3View.navigate('#L0-list');assert.equal(e['list-sort'].value,'number');assert.equal(e['list-direction'].value,'asc');
    assert.deepEqual(e['list-body'].children.map(row=>Number(row.dataset.neuron)),Array.from({length:50},(_,n)=>n));
    await c.Round3View.navigate('#L27');const n=1939,pitch=parseFloat(e['grid-canvas'].style.width)/64,event={pointerType:'touch',clientX:(n%64+.5)*pitch,clientY:(Math.floor(n/64)+.5)*pitch};
    await e['grid-canvas'].onclick(event);assert.equal(app.location.hash,'#L27-N1939');
    assert.equal(e.card.children[0].textContent,c.Round3Reader.say(lang,'cardTitle',{layer:27,neuron:1939}));
    assert.equal(e.card.all(x=>x.className==='tokens strong-tokens')[0].children.reduce((sum,x)=>sum+Number(x.dataset.count),0),20);
    assert(e.card.all(x=>x.className==='numeric-details').every(x=>!x.open));
    // Initial restored values and deferred pageshow values use the same path.
    const restored=await build(undefined,{hash:'#L23-list',viewport,controls:{token:'how',layer:'27'}}),re=restored.elements;
    await re.locale.onchange({target:{value:lang}});assert.equal(countFrom(re),expected.length);assert(!re['active-filters'].hidden);assert(re['active-filters'].textContent.includes('how'));
    re.token.value='';re.layer.value='27';re.language.value='ko';await restored.handlers.pageshow({persisted:true});
    const cat=unzip('catalog/L23.bin.gz'),schema=JSON.parse(fs.readFileSync(path.join(pages,'data/round3/stats/schema.json'))),ko=schema.languages.indexOf('ko');
    let expectedKo=0;for(let n=0;n<restored.context.Round3.WIDTHS[23];n++)expectedKo+=cat.readUInt8(n*12+8)===ko;
    assert.equal(countFrom(re),expectedKo);assert(re['active-filters'].textContent.includes('ko'));assert(!re['active-filters'].textContent.includes('how'));
    re.language.value='';await restored.handlers.pageshow({persisted:false});assert.equal(countFrom(re),restored.context.Round3.WIDTHS[23]);assert(re['active-filters'].hidden);
    restores+=3;cases++;totals.push({language:lang,viewport,word_matches:expected.length,language_matches:expectedKo});
  }
  console.log(JSON.stringify({passed:true,cases,totals,search_layer_isolated:true,word_filter_matches_index:true,clear_restores_both_widths:true,all_filter_names_visible:true,empty_state_clear:true,clear_recomputes_search:true,restore_checks:restores,sort_tap_cards_bilingual:true,autocomplete_off:true,phone_filter_wrap:true,viewport_measurement:'DOM geometry model; not a browser measurement',context_text_printed:false}));
}
main().catch(error=>{console.error(JSON.stringify({passed:false,frames:String(error.stack).split('\n').filter(s=>s.trim().startsWith('at ')).map(s=>s.match(/:(\d+):(\d+)/)?.[0]).filter(Boolean),context_text_printed:false}));process.exitCode=1;});
