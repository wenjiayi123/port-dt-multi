'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),test=require('node:test'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.resolve(__dirname,'../../app/ui/index.html'),'utf8');
const anchor=html.indexOf("  let masScenario = 'replay';");
assert(anchor>=0,'MAS controller exists');
const open=html.lastIndexOf('<script>',anchor),close=html.indexOf('</script>',anchor);
assert(open>=0 && close>anchor,'MAS controller has exact script boundaries');
const code=html.slice(open+'<script>'.length,close);
assert(code.includes("pullOverview('replay');"),'run actual controller including its initial load');
function element(){return {textContent:'',innerHTML:'',disabled:false,title:'',attrs:{},handlers:{},
 setAttribute(k,v){this.attrs[k]=v},getBoundingClientRect(){return {width:640,height:300}},
 addEventListener(k,fn){this.handlers[k]=fn},fire(){return this.handlers.click?.()}};}
function setup(){
 const nodes=new Map(),charts=new Map(),requests=[];
 const byId=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
 const context=vm.createContext({document:{querySelector:s=>byId(s.slice(1)),getElementById:byId},window:{addEventListener(){}},
 console:{warn(){}},requestAnimationFrame(){throw Error('visible chart should not defer')},
 echarts:{init(el){const chart={option:null,clears:0,setOption(value){this.option=value},clear(){this.option=null;this.clears++},resize(){}};charts.set(el,chart);return chart}},
 fetch:url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject}))});
 vm.runInContext(code,context);return {byId,charts,requests};
}
const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve()};
const payload=(scenario,n=10)=>({available:true,scenario:{id:scenario,label:scenario},kpis:{throughput_teu:n,delay_index_mean:n,peak_kw:n,energy_kwh:n,carbon_kg:n},
 agents:{qc:[{id:scenario}]},graph:{nodes:[{id:scenario,category:'qc'}],edges:[]},timeline:{categories:['qc'],items:[{name:scenario,category:'qc',start:0,end:2}]},
 conflicts:[],evidence:{dataset_id:scenario,dataset_sha256:scenario},decision:{algorithm:'SAC',job_id:scenario},site_replacement:{status:'simulation_only'}});
const reply=(request,data)=>request.resolve({ok:true,json:async()=>data});
const kpis=['teu','wp95','kw','kwh','co2'];
test('MAS initial and scenario loads clear old KPI, charts, plan and disable evidence until current response',async()=>{
 const {byId,charts,requests}=setup();assert.equal(requests.length,1);
 assert.match(byId('mas-plan-meta').textContent,/正在读取公开回放/);
 assert.equal(byId('mas-section').attrs['aria-busy'],'true');assert.equal(byId('mas-dispatch').disabled,true);
 reply(requests[0],payload('replay'));await flush();await byId('mas-dispatch').fire();assert.match(byId('mas-plan-meta').textContent,/证据：replay/);
 const pending=byId('mas-sim').fire();assert.match(requests[1].url,/scenario=dense$/);
 assert.match(byId('mas-plan-meta').textContent,/正在读取高密压测/);
 for(const id of kpis)assert.equal(byId('mas-kpi-'+id).textContent,'读取中');
 assert.equal(byId('mas-dispatch').disabled,true);for(const chart of charts.values())assert.equal(chart.option,null);
 const status=byId('mas-plan-meta').textContent;await byId('mas-dispatch').fire();assert.equal(byId('mas-plan-meta').textContent,status);assert.equal(requests.length,2);
 reply(requests[1],payload('dense',22));await pending;
 assert.equal(byId('mas-kpi-teu').textContent,'22');assert.equal(byId('mas-dispatch').disabled,false);assert.equal(byId('mas-section').attrs['aria-busy'],'false');
 await byId('mas-dispatch').fire();assert.match(byId('mas-plan-meta').textContent,/证据：dense/);
});
test('MAS rapid scenario changes allow only the newest response to render or enable evidence',async()=>{
 const {byId,charts,requests}=setup();const dense=byId('mas-sim').fire(),degraded=byId('mas-sim').fire();
 assert.match(requests[1].url,/scenario=dense$/);assert.match(requests[2].url,/scenario=degraded$/);
 requests[1].reject(Error('late dense failure'));await dense;
 assert.equal(byId('mas-section').attrs['aria-busy'],'true');assert.match(byId('mas-plan-meta').textContent,/正在读取降级压测/);
 reply(requests[2],payload('degraded',33));await degraded;const status=byId('mas-plan-meta').textContent;
 reply(requests[0],payload('replay',99));await flush();assert.equal(byId('mas-plan-meta').textContent,status);assert.equal(byId('mas-kpi-teu').textContent,'33');
 assert.equal(charts.get(byId('mas-graph')).option.series[0].data[0].id,'degraded');
 await byId('mas-dispatch').fire();assert.match(byId('mas-plan-meta').textContent,/证据：degraded/);
});
test('MAS current HTTP failure clears evidence and retry retains the chosen scenario',async()=>{
 const {byId,charts,requests}=setup();reply(requests[0],payload('replay'));await flush();
 const pending=byId('mas-sim').fire();requests[1].resolve({ok:false,text:async()=> 'HTTP 503 fixture'});await pending;
 assert.match(byId('mas-plan-meta').textContent,/高密压测证据读取失败.*503/);for(const id of kpis)assert.equal(byId('mas-kpi-'+id).textContent,'—');
 for(const chart of charts.values())assert.equal(chart.option,null);assert.equal(byId('mas-dispatch').disabled,true);assert.equal(byId('mas-section').attrs['aria-busy'],'false');
 const status=byId('mas-plan-meta').textContent;await byId('mas-dispatch').fire();assert.equal(byId('mas-plan-meta').textContent,status);assert.equal(requests.length,2);
 const retry=byId('mas-refresh').fire();assert.match(requests[2].url,/scenario=dense$/);reply(requests[2],payload('dense',44));await retry;
 assert.equal(byId('mas-kpi-teu').textContent,'44');assert.equal(byId('mas-dispatch').disabled,false);
 const propose=byId('mas-propose').fire();assert.match(requests[3].url,/scenario=replay$/);reply(requests[3],payload('replay',55));await propose;assert.equal(byId('mas-kpi-teu').textContent,'55');
});
test('MAS unavailable or mismatched scenario payload cannot become decision evidence',async()=>{
 for(const data of [{available:false},payload('dense')]){
  const {byId,requests}=setup();reply(requests[0],data);await flush();assert.equal(byId('mas-dispatch').disabled,true);assert.match(byId('mas-plan-meta').textContent,/证据读取失败/);
  for(const id of kpis)assert.equal(byId('mas-kpi-'+id).textContent,'—');await byId('mas-dispatch').fire();assert.equal(requests.length,1);
 }
});
