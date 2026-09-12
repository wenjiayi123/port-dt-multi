'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const root=path.resolve(__dirname,'../..'),html=fs.readFileSync(path.join(root,'app/ui/index.html'),'utf8');
const loading=fs.readFileSync(path.join(root,'app/static/module_loading.js'),'utf8');
const recovery=fs.readFileSync(path.join(root,'app/ui/adapters/runtime_recovery.js'),'utf8');
const flush=async()=>{for(let i=0;i<18;i++)await Promise.resolve();await new Promise(resolve=>setImmediate(resolve));};
function fixture({recover=false}={}){
 const nodes=new Map(),timers=new Map(),events=new Map(),docEvents=new Map(),requests=[];let timerId=0,current='home-hero';
 function el(){let id='',text='',markup='';const node={children:[],attrs:{},dataset:{},style:{removeProperty(k){delete this[k]}},hidden:false,disabled:false,handlers:{},
  classList:{add(){},remove(){},toggle(){}},setAttribute(k,v){this.attrs[k]=String(v)},removeAttribute(k){delete this.attrs[k]},
  addEventListener(k,fn){(this.handlers[k]??=[]).push(fn)},async click(){for(const fn of this.handlers.click||[])await fn({preventDefault(){}})},
  appendChild(child){this.children.push(child);return child},insertAdjacentElement(_where,child){this.children.push(child)},
  querySelector(selector){return (this.queries??={})[selector]??=((this.queries??={})[selector]=el())},querySelectorAll(){return[]},
  getClientRects(){return [{}]},getBoundingClientRect(){return {top:0,bottom:300,width:600,height:300}}};
  Object.defineProperties(node,{id:{get:()=>id,set:value=>{id=value;nodes.set(value,node)}},textContent:{get:()=>text,set:value=>{text=String(value);markup='';node.children=[]}},innerHTML:{get:()=>markup,set:value=>{markup=String(value);text='';node.children=[]}}});return node;
 }
 const byId=id=>{if(!nodes.has(id)){const node=el();node.id=id;}return nodes.get(id)};
 const listen=(map,name,fn)=>{if(!map.has(name))map.set(name,[]);map.get(name).push(fn)};
 const context={console:{warn(){}},URL,Headers,Response,Promise,Date,Map,Set,performance:{now:()=>100},AbortController,
  location:{href:'http://unit.test/',origin:'http://unit.test',hash:'',reload(){context.reloads++}},reloads:0,innerHeight:800,scrollY:0,
  setTimeout(fn,delay=0){const id=++timerId;timers.set(id,{fn,delay});return id},clearTimeout(id){timers.delete(id)},
  setInterval(){return ++timerId},clearInterval(){},requestAnimationFrame(fn){return context.setTimeout(fn,0)},
  addEventListener:(name,fn)=>listen(events,name,fn),dispatchEvent(event){for(const fn of events.get(event.type)||[])fn(event)},
  CustomEvent:class{constructor(type,options={}){this.type=type;this.detail=options.detail}},
  document:{readyState:'loading',hidden:false,head:el(),body:el(),getElementById:id=>nodes.get(id)||null,querySelector:()=>null,
   createElement:el,addEventListener:(name,fn)=>listen(docEvents,name,fn)},
  PortModuleNavigation:{get current(){return current},open(){throw Error('loader must never navigate')}},
  scrollTo(){throw Error('loader must never change page position')},
  fetch(url,init){if(url==='/health/live')return Promise.resolve(new Response('{"status":"alive"}'));return new Promise((resolve,reject)=>requests.push({url:String(url),init,resolve,reject}));}};
 context.window=context;vm.createContext(context);vm.runInContext(loading,context);
 const energy=[['b','yl-section','b-v3-status'],['c','hvac-section','c-v3-status'],['d','sbess-section','d-v3-status'],['be','be-section','be-v3-status'],['f','yc-section','f-v3-status']];
 for(const [,section,status] of energy){byId(section);byId(status)}
 byId('system-provenance-badge');
 if(recover)vm.runInContext(recovery,context);
 async function drain(){for(let i=0;i<8;i++){const zero=[...timers].filter(([,item])=>item.delay===0);for(const [id,item] of zero){timers.delete(id);item.fn()}await flush();}}
 async function ready(){context.document.readyState='interactive';for(const fn of docEvents.get('DOMContentLoaded')||[])fn();await drain();}
 async function show(id){current=id;context.document.body.dataset.moduleView=id;context.dispatchEvent(new context.CustomEvent('port-module-shown',{detail:{id}}));await drain();}
 const reply=(request,data)=>request.resolve(new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}}));
 return {context,requests,byId,nodes,energy,ready,show,drain,reply,events};
}
test('actual energy startup registers idle pages; homepage does not request or complete any of the five',async()=>{
 const f=fixture({recover:true}),{context:c,requests,byId}=f,calls=[];
 const names=['loadYardLightingV3','loadHvacV3','loadShoreBessV3','loadBessEnergyV3','loadYardCraneV3'];
 c.$=selector=>byId(selector.slice(1));c.handleYardLightingLoadFailure=()=>{};
 names.forEach((name,index)=>{c[name]=()=>c.__portDtTrackEvidenceLoad(f.energy[index][0],async()=>{calls.push(name);const r=await c.fetch('/api/energy/'+name);return r.json();});
  const line=html.split('\n').find(row=>row.includes('window.__portDtLoadWhenActive(')&&row.includes('()=>'+name+'().catch'));
  assert(line,'actual startup registration '+name);vm.runInContext(line,c);
 });
 assert.equal(calls.length,0);await f.ready();assert.equal(calls.length,0);assert.equal(requests.length,0);
 assert(c.PortModuleLoading.snapshot().every(row=>row.state==='idle'));
 assert.equal(byId('port-dt-runtime-recovery').hidden,true);
 for(const [key]of f.energy)assert.equal(byId('port-dt-progress-'+key).dataset.state,'idle');
 for(let i=0;i<names.length;i++){
  await f.show(f.energy[i][1]);assert.equal(calls.length,i+1);assert.equal(byId('port-dt-progress-'+f.energy[i][0]).dataset.state,'loading');
  assert.match(byId('port-dt-runtime-recovery').querySelector('small').textContent,/0\/1/);
  f.reply(requests[i],{sha256:'actual-response-'+i});await f.drain();
  assert.equal(byId('port-dt-progress-'+f.energy[i][0]).dataset.state,'complete');
  assert.match(byId('port-dt-runtime-recovery').querySelector('small').textContent,/1\/1/);
 }
 await f.show('home-hero');assert.equal(calls.length,5);assert.equal(byId('port-dt-runtime-recovery').hidden,true);assert.equal(c.location.hash,'');
});
test('late energy completion never reopens its page; in-flight tracker calls share one real task and failures remain failed',async()=>{
 const f=fixture({recover:true}),c=f.context;await f.ready();await f.show('sbess-section');let resolve,calls=0;
 const task=()=>{calls++;return new Promise(r=>resolve=r)};
 const a=c.__portDtTrackEvidenceLoad('d',task),b=c.__portDtTrackEvidenceLoad('d',task);await flush();assert.equal(calls,1);
 await f.show('home-hero');resolve({model_sha256:'fixture'});await Promise.all([a,b]);await f.drain();assert.equal(c.PortModuleLoading.current(),'home-hero');assert.equal(f.byId('port-dt-runtime-recovery').hidden,true);
 await assert.rejects(c.__portDtTrackEvidenceLoad('f',()=>Promise.reject(Error('SHA mismatch'))),/SHA mismatch/);
 assert.equal(f.byId('port-dt-progress-f').dataset.state,'failed');assert.notEqual(f.byId('port-dt-progress-f').dataset.state,'complete');
});
test('identical API GETs coalesce only while pending and defer heavy requests until routing; writes/signals/query changes stay independent',async()=>{
 const f=fixture(),c=f.context;
 const a=c.fetch('/api/model?seed=1',{cache:'no-store'}),b=c.fetch('/api/model?seed=1',{cache:'no-store'});
 await flush();assert.equal(f.requests.length,0,'heavy GET cannot occupy connections before DOM route initialization');
 const summary=c.fetch('/api/system/provenance?detail=summary',{cache:'no-store'});await flush();assert.equal(f.requests.length,1);f.reply(f.requests[0],{detail_level:'summary'});await summary;
 await f.ready();assert.equal(f.requests.length,2);f.reply(f.requests[1],{model_sha256:'v1'});
 const [ra,rb]=await Promise.all([a,b]);assert.notEqual(ra,rb);assert.deepEqual(await ra.json(),{model_sha256:'v1'});assert.deepEqual(await rb.json(),{model_sha256:'v1'});
 const fresh=c.fetch('/api/model?seed=1',{cache:'no-store'});await flush();assert.equal(f.requests.length,3,'completed model response is not reused across hash checks');f.reply(f.requests[2],{model_sha256:'v2'});assert.equal((await (await fresh).json()).model_sha256,'v2');
 const independent=[c.fetch('/api/model?seed=2'),c.fetch('/api/write',{method:'POST',body:'{}'}),c.fetch('/api/write',{method:'POST',body:'{}'}),c.fetch('/api/model',{signal:new AbortController().signal}),c.fetch('/api/model',{signal:new AbortController().signal})];await flush();assert.equal(f.requests.length,8);for(const r of f.requests.slice(3))f.reply(r,{});await Promise.all(independent);
});
test('page load failures can retry, repeated show shares pending work, and current page state is never restored from storage',async()=>{
 const f=fixture(),c=f.context;let calls=0,reject,resolve;
 c.__portDtLoadWhenActive('mas-section',()=>{calls++;return new Promise((yes,no)=>{resolve=yes;reject=no})},{key:'mas'});
 await f.ready();assert.equal(calls,0);assert.equal(c.PortModuleLoading.current(),'home-hero');
 await f.show('mas-section');await f.show('mas-section');assert.equal(calls,1);reject(Error('offline'));await f.drain();assert.equal(c.PortModuleLoading.snapshot()[0].state,'failed');
 await f.show('mas-section');assert.equal(calls,2);resolve({sha:'verified'});await f.drain();await f.show('home-hero');await f.show('mas-section');assert.equal(calls,2,'return can reuse current page evidence until explicit refresh or expiry');
 const refresh=c.PortModuleLoading.refresh('mas');await flush();assert.equal(calls,3);resolve({sha:'new'});await refresh;await f.show('home-hero');assert.equal(c.location.hash,'');
});
for(const spec of [
 {section:'ext-section',anchor:'const start=()=>Promise.all([!externalV3Evidence?',loads:['loadExternalV3','loadPortCallReadiness','loadSiteIntegrationReadiness'],empty:['externalV3Evidence','portCallReadiness','siteIntegrationReadiness'],status:'ext-v3-status'},
 {section:'g-section',anchor:'const start=()=>governance?',loads:['load'],empty:['governance'],status:'gov-detail'},
 {section:'opsx-section',anchor:'const start=()=>opsxV3Evidence?',loads:['loadOpsXV3'],empty:['opsxV3Evidence'],status:'opsx-v3-status'},
 {section:'mlops-section',anchor:'const start=()=>evidence?',loads:['load'],empty:['evidence'],status:'mlops-status'},
])test(`actual ${spec.section} startup waits for its requests and retries a failed first visit without the reuse delay`,async()=>{
 const f=fixture(),c=f.context,pending=[];c.byId=f.byId;c.governanceError=error=>{f.byId(spec.status).textContent='治理证据加载失败：'+error.message};
 for(const key of spec.empty)c[key]=null;c.portCallInteraction=0;c.siteIntegrationInteraction=0;
 for(const name of spec.loads)c[name]=()=>new Promise((resolve,reject)=>pending.push({name,resolve,reject}));
 const start=html.indexOf(spec.anchor);assert(start>=0,`actual ${spec.section} loader`);
 const registration=`window.__portDtLoadWhenActive('${spec.section}',start);`,end=html.indexOf(registration,start);assert(end>start);
 vm.runInContext(html.slice(start,end+registration.length),c);
 await f.ready();assert.equal(pending.length,0);await f.show(spec.section);assert.equal(pending.length,spec.loads.length);
 assert.equal(c.PortModuleLoading.snapshot()[0].state,'loading','startup must await real requests');
 // A multi-request page must remain loading even after one endpoint succeeds.
 if(pending.length>1){pending.at(-1).resolve({ready:true});await f.drain();assert.equal(c.PortModuleLoading.snapshot()[0].state,'loading');}
 pending[0].reject(Error('first visit failed'));for(const p of pending.slice(1))p.resolve({ready:true});await f.drain();
 assert.equal(c.PortModuleLoading.snapshot()[0].state,'failed');assert.match(f.byId(spec.status).textContent,/first visit failed/);
 await f.show('home-hero');await f.show(spec.section);assert.equal(pending.length,spec.loads.length*2,'failed visits bypass the 60s successful-result reuse');
 assert.equal(c.PortModuleLoading.snapshot()[0].state,'loading');for(const p of pending.slice(spec.loads.length))p.resolve({ready:true});await f.drain();
 assert.equal(c.PortModuleLoading.snapshot()[0].state,'loaded');
});
function registry(f){
 const start=html.indexOf('const contractState ='),end=html.indexOf('const chainState =',start);assert(start>=0&&end>start);
 f.context.$=f.byId;f.context.document.getElementById=f.byId;
 vm.runInContext(html.slice(start,end)+';this.registry={state:contractState,load:tryLoadContractRegistry,render:renderContractRegistry,detail:loadContractDetail};',f.context);
 return f.context.registry;
}
const summaryPayload={rl:{runtime:{available:null},datasets:null,verification_state:'deferred'},runtime_policy:{available:null,verification_state:'deferred',production_authority:false},telemetry:{mode:'calibrated_public_replay_simulator',measured:false},feature_flags:{},external_adapters:{}};
test('actual homepage registry summary leaves RL and policy unrequested and displays deferred rather than zero/unavailable',async()=>{
 const f=fixture();await f.ready();const r=registry(f),pending=r.load();await flush();assert.equal(f.requests.length,2);
 const summary=f.requests.find(row=>row.url.includes('provenance'));assert.equal(summary.url,'/api/system/provenance?detail=summary');f.reply(summary,summaryPayload);f.reply(f.requests.find(row=>row.url.includes('actuators')),{enabled:false});r.state.items=await pending;r.state.selected='rl';r.render();
 for(const key of ['rl','runtime-policy']){const item=r.state.items.find(row=>row.key===key);assert.equal(item.verification_state,'deferred');assert.equal(item.readiness,'按需核验');assert.equal(item.mode,'按需核验');assert.equal(item.ready,false);assert.doesNotMatch(item.summary,/数据集=0|模型=不可用/);}
 assert.equal(f.requests.length,2,'initial selection does not fetch full capabilities/model');assert.match(f.byId('contract-detail').innerHTML,/尚未请求/);
});
test('actual registry detail clicks coalesce; late other-card success cannot overwrite selected failure and retry shows original JSON',async()=>{
 const f=fixture();await f.ready();const r=registry(f);r.state.items=[{key:'rl',name:'RL',detail_url:'/api/rl/engine/capabilities',verification_state:'deferred'},{key:'runtime-policy',name:'Policy',detail_url:'/api/v3/runtime/status',verification_state:'deferred'}];r.state.selected='rl';r.render();
 const rl=r.detail('rl');await flush();assert.equal(r.state.items[0].readiness,'核验中');const dup=r.detail('rl');await flush();assert.equal(f.requests.length,1);
 r.state.selected='runtime-policy';const policy=r.detail('runtime-policy');await flush();f.reply(f.requests[0],{engine:'fixture-engine',runtime:{available:true},algorithms:[{name:'SAC'}],datasets:[{sha256:'verified-sha'}]});await Promise.all([rl,dup]);assert.match(f.byId('contract-detail').innerHTML,/正在请求完整/);assert.doesNotMatch(f.byId('contract-detail').innerHTML,/fixture-engine/);
 const completedCard=f.byId('contract-grid').children.find(card=>card.dataset.contractKey==='rl');assert.match(completedCard.innerHTML,/runtime-ready/);assert.doesNotMatch(completedCard.innerHTML,/核验中/);
 f.requests[1].resolve(new Response('unavailable',{status:503}));await policy;assert.match(f.byId('contract-detail').innerHTML,/核验失败|读取失败/);assert.equal(r.state.items[1].ready,false);
 const retry=r.detail('runtime-policy');await flush();f.reply(f.requests[2],{available:true,inference:'real-loaded-actor',model:{algorithm:'SAC',model_sha256:'verified-model'},production_authority:false});await retry;
 assert.equal(r.state.items[1].mode,'real-loaded-actor');const pre=f.byId('contract-detail').children.at(-1);assert.match(pre.style.cssText,/max-height:360px;overflow:auto/);assert.equal(JSON.parse(pre.textContent).model.model_sha256,'verified-model');assert.equal(JSON.parse(pre.textContent).production_authority,false);
});
