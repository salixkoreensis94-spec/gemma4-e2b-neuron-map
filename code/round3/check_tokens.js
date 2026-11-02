/* Numeric-only report. Token values and evidence text are never printed. */
'use strict';
const fs=require('fs'),path=require('path'),vm=require('vm'),zlib=require('zlib'),assert=require('assert');
const root=path.resolve(__dirname,'../..'),stats=path.join(root,'docs/map/data/round3/stats'),context={TextEncoder};context.globalThis=context;vm.createContext(context);
for(const file of ['decoder.js','reader.js'])vm.runInContext(fs.readFileSync(path.join(root,'code/round3',file),'utf8'),context);
const Q=context.Round3Reader,R=context.Round3,vocab={};for(const file of fs.readdirSync(path.join(stats,'vocab')))Object.assign(vocab,JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(stats,'vocab',file)))));
function raw(id){assert(Object.hasOwn(vocab,id));return vocab[id];}
function nonempty(value){assert(value.trim().length>0);assert(!/[\p{Cc}\p{Cf}\p{Cs}\p{Default_Ignorable_Code_Point}]/u.test(value));}
function hasLetter(value){if(value==='[withheld token]'||value.includes('\ufffd'))return false;return [...value].some(ch=>/\p{L}/u.test(ch)&&/[\p{Script=Hangul}\p{Script=Latin}\p{Script=Han}]/u.test(ch)&&! /\p{Default_Ignorable_Code_Point}/u.test(ch));}
const cards=new Map(),lists=new Map();function card(layer,neuron){const key=`L${String(layer).padStart(2,'0')}_${String(Math.floor(neuron/256)).padStart(2,'0')}.bin.gz`;if(!cards.has(key)){const b=zlib.gunzipSync(fs.readFileSync(path.join(stats,'cards',key)));cards.set(key,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}return R.decodeCard(cards.get(key),layer,neuron);}
function list(layer,neuron){if(!lists.has(layer))lists.set(layer,zlib.gunzipSync(fs.readFileSync(path.join(stats,'browse',`L${String(layer).padStart(2,'0')}.bin.gz`))));return Array.from({length:6},(_,i)=>lists.get(layer).readUInt32LE(neuron*25+i*4)).filter(id=>id!==0xffffffff);}
let labelChecks=0;function checkNeuron(layer,neuron){const c=card(layer,neuron);for(const lang of ['ko','en']){for(const id of [...c.tokenIds,...c.fireIds,...c.topIds,...c.bottomIds,...list(layer,neuron)]){nonempty(Q.piece(raw(id),lang));labelChecks++;}nonempty(Q.summaryTokens(c.tokenIds,raw,lang));}}
function main(){
  let fixtures=0;
  for(let cp=0;cp<=0x9f;cp++)if(cp<32||cp>=0x7f){const value=Q.visible(String.fromCodePoint(cp));nonempty(value);if(cp<32&&cp!==9&&cp!==10)assert(value===String.fromCodePoint(0x2400+cp));if(cp===127)assert(value==='␡');if(cp>=128)assert(value===`⟦U+${cp.toString(16).toUpperCase().padStart(4,'0')}⟧`);fixtures++;}
  for(const cp of [0x00a0,0x00ad,0x034f,0x061c,0x115f,0x1160,0x1680,0x17b4,0x180b,0x180e,0x2000,0x2001,0x2002,0x2003,0x2004,0x2005,0x2006,0x2007,0x2008,0x2009,0x200a,0x200b,0x200c,0x200d,0x200e,0x200f,0x2028,0x2029,0x202a,0x202b,0x202c,0x202d,0x202e,0x202f,0x205f,0x2060,0x2066,0x2067,0x2068,0x2069,0x2800,0x3000,0x3164,0xfe0f,0xfeff,0xffa0,0xe0001,0xe0020,0xe0100]){assert(Q.visible(String.fromCodePoint(cp))===`⟦U+${cp.toString(16).toUpperCase().padStart(4,'0')}⟧`);fixtures++;}
  assert(Q.visible('\r')==='␍');assert(Q.visible(' \n\t')==='␣↵⇥');assert(Q.visible('')==='⟦∅⟧');assert(Q.visible('A한字')==='A한字');
  const starting=JSON.parse(fs.readFileSync(path.join(stats,'starting.json')));assert(starting.items.length===8);let special=0;
  for(const row of starting.items){row.tokens.forEach(value=>{nonempty(Q.visible(value));for(const lang of ['ko','en'])nonempty(Q.piece(value,lang));});checkNeuron(row.layer,row.neuron);if(row.layer===27&&row.neuron===1939){special++;continue;}const c=card(row.layer,row.neuron);assert(hasLetter(raw(c.tokenIds[0])));assert(c.tokenIds.filter(id=>hasLetter(raw(id))).length>=10);assert(![[1,2447],[11,3562],[34,4816]].some(([l,n])=>l===row.layer&&n===row.neuron));}
  assert(special===1);
  let seed=2026110302;const ids=new Set();while(ids.size<1000){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;ids.add((seed>>>0)%337920);}for(const id of ids){const [layer,neuron]=R.locate(id);checkNeuron(layer,neuron);}
  // The original carriage-return case remains readable even after leaving the start set.
  const cr=card(1,2447);assert(Q.piece(raw(Q.strong(cr.tokenIds)[0].id),'ko')==='␍');
  console.log(JSON.stringify({passed:true,seed:2026110302,random_neurons:1000,starting_points:8,formatted_labels_checked:labelChecks,control_fixtures:fixtures,carriage_return_case:{layer:1,neuron:2447},letter_gate:true,summary_and_list_nonempty:true,context_text_printed:false}));
}
try{main();}catch(error){console.error(JSON.stringify({passed:false,frames:String(error.stack).split('\n').filter(s=>s.trim().startsWith('at ')).map(s=>s.match(/:(\d+):(\d+)/)?.[0]).filter(Boolean),context_text_printed:false}));process.exitCode=1;}
