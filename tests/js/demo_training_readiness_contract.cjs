// Exercise the real guide with delayed readiness events and controlled timers.
// This is a unit fixture, not browser or model-training evidence.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
function setup(){
 const events={},timers=new Map();let serial=0,ready=false,clicks=0;
 const button={disabled:false,getClientRects:()=>[1],scrollIntoView(){},click(){clicks++;},classList:{add(){},remove(){}}};
 const dataset={getAttribute:()=>String(ready)};
 const document={readyState:'loading',head:{append(){}},createElement:()=>({}),addEventListener(){},
  querySelector:s=>s==='#selDataset'?dataset:null,
  querySelectorAll:s=>s==='#btnStartTrain'?[button]:[]};
 const store=new Map([['port_dt_native_demo_guide_v1',JSON.stringify({open:true,current:7,phase:'ready',completed:[],skipped:[]})]]);
 const context={document,console,Date,JSON,Array,Number,String,Set,sessionStorage:{getItem:k=>store.get(k),setItem:(k,v)=>store.set(k,v)}};
 context.window={location:{pathname:'/rl-panel'},getComputedStyle:()=>({display:'block',visibility:'visible'}),
  addEventListener:(name,fn)=>events[name]=fn,
  setTimeout:(fn,ms)=>{const id=++serial;timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id)};
 vm.createContext(context);
 const source=fs.readFileSync('app/ui/adapters/demo_guide.js','utf8').replace(/\}\)\(\);\s*$/,'window.fixtureGuide={runCurrent,jumpToStep,state:()=>state};})();');
 vm.runInContext(source,context);
 return{guide:context.window.fixtureGuide,timers,events,clicks:()=>clicks,ready:value=>ready=value,
  tick(ms){for(const [id,t]of [...timers])if(t.ms===ms){timers.delete(id);t.fn();}}};
}
test('one guide action waits for verified dataset and profile, then opens confirmation exactly once',()=>{
 const h=setup();h.guide.runCurrent();assert.equal(h.clicks(),0);assert.equal(h.guide.state().phase,'waiting');
 h.ready(true);h.events['port-dt:training-contract-ready']();h.tick(180);assert.equal(h.clicks(),1);
 h.events['port-dt:training-contract-ready']();h.tick(180);assert.equal(h.clicks(),1);
});
test('changing guide steps revokes a pending readiness action',()=>{
 const h=setup();h.guide.runCurrent();h.guide.jumpToStep(6,'test navigation');
 h.ready(true);h.events['port-dt:training-contract-ready']();h.tick(180);assert.equal(h.clicks(),0);
});
test('missing readiness unlocks with an error and never opens or starts training',()=>{
 const h=setup();h.guide.runCurrent();h.tick(60000);assert.equal(h.guide.state().phase,'error');assert.equal(h.clicks(),0);
 h.ready(true);h.events['port-dt:training-contract-ready']();h.tick(180);assert.equal(h.clicks(),0);
});
