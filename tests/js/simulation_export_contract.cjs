// Execute actual chart/export handlers with deterministic canvas/DOM fixtures.
// This is a state-and-drawing contract, not a screenshot or browser download test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const html=fs.readFileSync('app/ui/index.html','utf8');
function part(start,end){const a=html.indexOf(start),b=html.indexOf(end,a+start.length);assert(a>=0&&b>a);return html.slice(a,b)}
function harness(){
 const nodes=new Map(),downloads=[],texts=[],drawing=[],messages=[];
 const context=new Proxy({createLinearGradient:()=>({addColorStop(){}}),fillText:(text)=>texts.push(text),lineTo:(...args)=>drawing.push(args)},{get(target,key){return key in target?target[key]:()=>{}},set(target,key,value){target[key]=value;return true}});
 function element(id){return {id,value:'',textContent:'',innerHTML:'',dataset:{},disabled:false,style:{},width:640,height:180,handlers:{},selectedOptions:[],
  getContext:()=>context,getBoundingClientRect:()=>({width:640,height:180}),toDataURL:()=>{drawing.push('toDataURL');return'data:image/png;base64,fixture'},
  addEventListener(name,fn){(this.handlers[name]??=[]).push(fn)},click(){downloads.push(this.download)},setAttribute(){}}}
 const node=id=>{id=id.replace(/^#/,'');if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id)};
 node('rlpanel-strategy').value='model-a';node('rlpanel-strategy').dataset.payload='[{"id":"model-a"},{"id":"model-b"}]';
 const document={getElementById:node,createElement:element};
 const c=vm.createContext({document,$:node,window:{location:{origin:'http://unit.test'},selAsset:{value:'WRONG-OLD-ASSET'}},Date,Math,Number,Array,String,JSON,devicePixelRatio:1,
  show:message=>messages.push(message),renderStrategyDecisionChain(){},updateExecSid(){},calibStats:{available:false},fetch:async()=>({ok:true,json:async()=>({strategy_id:'model-a',baseline:{agg_kW:[200,220,210]},simulated:{agg_kW:[190,205,215]},summary:{}})})});
 vm.runInContext(part("    const btnOpenRL = $('#btn-rlpanel-open');",'    function humanList('),c);
 vm.runInContext(part('    function drawSimChart(aggBase, aggSim){','    /************** 可解释特征'),c);
 vm.runInContext(part('    function saveCanvasPNG(id, filename){','    function ensureSimSaveButton(){'),c);
 vm.runInContext(part('    btnSimRL.onclick = async () => {','        // ====== 新增：仿真图保存'),c);
 vm.runInContext(part("    selRL?.addEventListener('change', ()=>{",'    function proxyTopNavAction('),c);
 vm.runInContext('drawSim()',c);
 return {c,node,downloads,texts,drawing,messages,run:code=>vm.runInContext(code,c)};
}
test('no evaluation cannot export an empty grid; valid current curves enable a labelled opaque image',async()=>{
 const h=harness();assert(h.node('btn-sim-save').disabled);h.c.saveCurrentSimulationPNG();assert.equal(h.downloads.length,0);assert.match(h.node('simMeta').textContent,/没有可保存/);
 await h.run('btnSimRL.onclick()');assert.equal(h.node('btn-sim-save').disabled,false);assert(h.drawing.some(item=>Array.isArray(item)&&item.every(Number.isFinite)));
 assert(h.texts.some(text=>text.includes('model-a')&&text.includes('kW')));h.c.saveCurrentSimulationPNG();assert.equal(h.downloads.length,1);assert.match(h.downloads[0],/^rl-sim-.*\.png$/);
});
test('missing, mismatched, or nonfinite curves never enable export and failed requests clear old data',async()=>{
 for(const bad of [{baseline:{agg_kW:[]},simulated:{agg_kW:[]}},{baseline:{agg_kW:[1,2]},simulated:{agg_kW:[1]}},{baseline:{agg_kW:[1,NaN]},simulated:{agg_kW:[1,2]}}]){
  const h=harness();await h.run('btnSimRL.onclick()');h.c.fetch=async()=>({ok:true,json:async()=>({strategy_id:'model-a',...bad})});await h.run('btnSimRL.onclick()');
  assert(h.node('btn-sim-save').disabled);assert.match(h.node('simMeta').textContent,/评测失败/);h.c.saveCurrentSimulationPNG();assert.equal(h.downloads.length,0);
 }
 const h=harness();await h.run('btnSimRL.onclick()');h.c.fetch=async()=>{throw Error('offline')};await h.run('btnSimRL.onclick()');assert(h.node('btn-sim-save').disabled);assert.equal(h.run('simState.base.length'),0);
});
test('reselection clears the image and a late response cannot reenable the old strategy export',async()=>{
 const h=harness();await h.run('btnSimRL.onclick()');let finish;
 h.c.fetch=()=>new Promise(resolve=>finish=resolve);const pending=h.run('btnSimRL.onclick()');assert(h.node('btn-sim-save').disabled);
 h.node('rlpanel-strategy').value='model-b';h.node('rlpanel-strategy').handlers.change[0]();
 finish({ok:true,json:async()=>({strategy_id:'model-a',baseline:{agg_kW:[1,2]},simulated:{agg_kW:[1,2]}})});await pending;
 assert(h.node('btn-sim-save').disabled);assert.equal(h.run('simState.base.length'),0);h.c.saveCurrentSimulationPNG();assert.equal(h.downloads.length,0);assert.equal(h.node('btn-rlpanel-sim').disabled,false);
});
test('left-panel image uses actual asset selection and marks missing forecast/actual calibration',()=>{
 const h=harness();h.node('asset').value='bess-01';h.node('asset').selectedOptions=[{textContent:'储能 BESS'}];h.node('title-asset').textContent='实时功率（储能 BESS）';h.node('dl-mape').textContent='1.0%';
 vm.runInContext(part('    function exportLeftPanelPNG(){','    function drawHeat('),h.c);h.c.exportLeftPanelPNG();
 assert(h.texts.some(text=>text.includes('储能 BESS [bess-01]')));assert(!h.texts.some(text=>text.includes('WRONG-OLD-ASSET')));
 assert(h.texts.some(text=>text.includes('实时功率（储能 BESS）')&&text.includes('预测实绩对齐数据待接入港口')));
 assert(h.texts.includes('MAPE：待接入港口'));assert.equal(h.downloads.length,1);
 h.c.calibStats.available=true;h.texts.length=0;h.c.exportLeftPanelPNG();assert(h.texts.includes('MAPE：1.0%'));
});
