// Delayed-response DOM contract fixtures, not browser click coverage.
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const html=fs.readFileSync(path.resolve(__dirname,'../../app/ui/index.html'),'utf8');
const script=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].find(m=>m[1].includes('let externalV3Evidence=null;'))[1];
const payload={dataset:{dataset_id:'fixture',rows:1},tables:{},signal_registry:[],timeline:[{fixture:true}],public_sources:['fixture-source'],adapter_status:{},boundary:{site_status:'closed'},replacement_contract:{required:true}};
function setup(){
  const nodes=new Map(),timers=[],requests=[];
  function node(id){if(!nodes.has(id))nodes.set(id,{textContent:'',innerHTML:'',style:{},listeners:[],addEventListener(type,fn,capture){this.listeners.push({type,fn,capture});},async click(){const results=this.listeners.filter(x=>x.type==='click').sort((a,b)=>Number(Boolean(b.capture))-Number(Boolean(a.capture))).map(x=>x.fn({preventDefault(){},stopImmediatePropagation(){}}));await Promise.all(results);},getClientRects(){return[];},getBoundingClientRect(){return{width:0,height:0};},querySelector(){return node(id+'-tbody');},appendChild(){}});return nodes.get(id);}
  const context={Date,JSON,Math,Number,String,Promise,console,setTimeout(fn,delay){timers.push({fn,delay});},document:{getElementById:node,createElement:()=>node('new-'+nodes.size)},window:{devicePixelRatio:1,addEventListener(){}},fetch(url){return new Promise((resolve,reject)=>requests.push({url,resolve,reject}));}};
  vm.createContext(context);vm.runInContext(script,context);
  const resolve=(url,data=payload)=>{const row=requests.find(x=>x.url===url&&!x.done);assert.ok(row,`pending request ${url}`);row.done=true;row.resolve({ok:true,json:async()=>data});};
  const settle=()=>new Promise(resolve=>setImmediate(resolve));
  return{node,timers,requests,resolve,settle,context};
}
(async()=>{
  let h=setup();const first=h.node('btn-ext-provenance').click();
  h.timers.find(x=>x.delay===1200).fn();h.timers.find(x=>x.delay===4000).fn();
  assert.equal(h.requests.filter(x=>x.url.includes('external-signals')).length,1,'click plus both delayed starts coalesce');
  h.resolve('/api/v3/external-signals/evidence');await first;
  assert.equal(JSON.parse(h.node('ext-v3-detail').textContent).dataset.dataset_id,'fixture');
  const refresh=h.node('btn-ext-refresh').click();h.resolve('/api/v3/external-signals/evidence',{...payload,dataset:{dataset_id:'new-fixture',rows:2}});await refresh;await h.settle();
  assert.equal(JSON.parse(h.node('ext-v3-detail').textContent).dataset.dataset_id,'new-fixture','refresh preserves selected provenance view');
  h.timers.find(x=>x.delay===4000).fn();assert.equal(h.requests.filter(x=>x.url.includes('external-signals')).length,2,'successful evidence is not reloaded by startup retry');
  h=setup();const provenance=h.node('btn-ext-provenance').click();const registry=h.node('btn-ext-registry').click();h.resolve('/api/v3/external-signals/evidence');await Promise.all([provenance,registry]);assert.ok(JSON.parse(h.node('ext-v3-detail').textContent).timeline_sample,'last selected view wins');
  h=setup();const failed=h.node('btn-ext-contract').click();h.requests[0].done=true;h.requests[0].reject(new Error('fixture offline'));await failed;assert.match(h.node('ext-v3-detail').textContent,/读取失败/);assert.match(h.node('ext-v3-status').textContent,/加载失败/);
  const retry=h.node('btn-ext-refresh').click();h.resolve('/api/v3/external-signals/evidence');await retry;await h.settle();assert.equal(JSON.parse(h.node('ext-v3-detail').textContent).replacement_contract.required,true);
  console.log('EXTERNAL_DETAIL_LOADING_CONTRACT:PASS:5 (delayed DOM fixtures; no browser claim)');
})().catch(error=>{console.error(error);process.exitCode=1;});
