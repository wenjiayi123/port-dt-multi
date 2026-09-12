// Executes the actual compliance controller with delayed HTTP fixtures.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.resolve(__dirname,'../../app/ui/index.html'),'utf8');
const script=html.split("const periodSeg = byId('cmp-period');")[1].split('// 外部环境：只读查询')[0];
function setup(){
 const nodes=new Map(),requests=[];
 const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',innerHTML:'',style:{},disabled:false,handlers:{},addEventListener(type,fn){(this.handlers[type]??=[]).push(fn);},querySelector(selector){return id==='cmp-period'?buttons.find(x=>x.active):node(id+'-tbody');},reportValidity(){return this.value!==''&&Number.isFinite(Number(this.value));},emit(type,event={}){this.handlers[type]?.forEach(fn=>fn(event));}});return nodes.get(id);};
 const buttons=['month','quarter'].map(p=>{const b={dataset:{p},tagName:'BUTTON',active:p==='month'};b.classList={add(){b.active=true;},remove(){b.active=false;}};return b;});
 for(const [id,value]of Object.entries({'cmp-month':'2025-10','cmp-teu':'12000','cmp-gran':'all','cmp-grid':'565','cmp-diesel':'2.68','cmp-selfkg':'0.70','cmp-share':'0','cmp-diesel-model':'rule_of_thumb'}))node(id).value=value;
 node('cmp-month').reportValidity=()=>/^\d{4}-\d{2}$/.test(node('cmp-month').value);
 const context={JSON,Date,String,Number,Promise,URLSearchParams,console,byId:node,$$:()=>buttons,nfmt:n=>Number(n).toFixed(2),fmt2:n=>Number(n).toFixed(2),pct:n=>String(n),show(){},fetch:url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject}))};
 vm.createContext(context);vm.runInContext("const periodSeg = byId('cmp-period');"+script,context);return{node,buttons,requests,context,run:code=>vm.runInContext(code,context)};
}
const result={totals:{electricity_kWh:3054474,scope1_kg:10,scope2_kg:300,total_kg:310},allocations:{}};
(async()=>{
 let h=setup();const first=h.run("runCompliance('GET')");h.requests[1].resolve({ok:true,json:async()=>result});await first;assert.match(h.node('cmp-elec').textContent,/3054474/);assert.match(h.node('cmp-result-scope').textContent,/2025-10-31/);
 h.node('cmp-period').emit('click',{target:h.buttons[1]});assert.match(h.node('cmp-range').textContent,/2025-12-31/);assert.equal(h.node('cmp-elec').textContent,'待重新计算');assert.match(h.node('cmp-result-scope').textContent,/已失效/);
 const second=h.run("runCompliance('POST')");assert.equal(h.node('btn-cmp-run').disabled,true);h.node('cmp-teu').value='13000';h.node('cmp-teu').emit('input');h.requests[2].resolve({ok:true,json:async()=>result});await second;assert.equal(h.node('cmp-elec').textContent,'待重新计算','late old-parameter response cannot overwrite changed inputs');assert.equal(h.node('btn-cmp-run').disabled,false);
 h.requests[0].resolve({ok:true,json:async()=>({evidence_mode:'public_data_offline_engineering',baseline_electricity_mwh:9999})});await new Promise(resolve=>setImmediate(resolve));assert.equal(h.node('cmp-elec').textContent,'待重新计算','initial reference cannot overwrite calculation/input changes');
 const failed=h.run("runCompliance('GET')");h.requests[3].reject(new Error('offline fixture'));await failed;for(const id of ['cmp-elec','cmp-kwh-teu','cmp-kg-teu'])assert.equal(h.node(id).textContent,'计算失败');assert.match(h.node('cmp-result-scope').textContent,/计算失败/);
 console.log('COMPLIANCE_RESULT_SCOPE_CONTRACT:PASS:5 (delayed DOM fixtures; no browser claim)');
})().catch(error=>{console.error(error);process.exitCode=1;});
