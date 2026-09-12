// Delayed DOM fixtures only; no browser or live API access.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.resolve(__dirname,'../../app/ui/index.html'),'utf8');
const script=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].find(m=>m[1].includes('function renderSelectedMlopsDetail()'))[1];
function setup(){
 const nodes=new Map(),timers=[],requests=[],loaders=[];
 const node=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',value:'',innerHTML:'',style:{},listeners:[],querySelector(){return node(id+'-tbody');},addEventListener(type,fn){this.listeners.push(fn);},async click(){await Promise.all(this.listeners.map(fn=>fn({stopImmediatePropagation(){}})));}});return nodes.get(id);};
 const context={window:{__portDtLoadWhenActive(section,task){assert.equal(section,'mlops-section');loaders.push(task);}},JSON,Number,String,Promise,console,setTimeout(fn,delay){timers.push({fn,delay});},document:{getElementById:node},fetch:url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject}))};vm.createContext(context);vm.runInContext(script,context);return{node,timers,requests,context,loaders};
}
const data={summary:{formal_runs:34,smoke_runs:32},evaluation:{selected_job_ids:['fixture-a','fixture-b'],selected_algorithm:'sac'},reproducibility:{dataset_id:'fixture-dataset',dataset_sha256:'a'.repeat(64)},pipeline:[{id:'quality',status:'pass',evidence:'quality evidence'},{id:'split',status:'pass',evidence:'split evidence'}],boundary:{production_authority:false},artifact_manifest:[],algorithms:[],replacement_and_rollback:{fixed:true}};
(async()=>{
 let h=setup();const detail=h.node('btn-mlops-registry').click();h.loaders.forEach(task=>task());assert.equal(h.requests.length,1);h.requests[0].resolve({ok:true,json:async()=>data});await detail;
 assert.equal(JSON.parse(h.node('mlops-detail').textContent).summary.formal_runs,34);h.loaders.forEach(task=>task());assert.equal(h.requests.length,1,'startup retry does not reset selected detail');assert.match(h.node('dq-res').textContent,/通过/);assert.equal(h.node('o11y-job').value,'fixture-a, fixture-b');assert.equal(h.node('dq-dataset').value,'fixture-dataset');
 h=setup();const blocked=h.node('btn-dq-check').click();h.requests[0].resolve({ok:true,json:async()=>({...data,pipeline:[{id:'quality',status:'pending'},{id:'split',status:'fail'}]})});await blocked;assert.match(h.node('dq-res').textContent,/未通过/);assert.equal(h.node('dq-res').style.color,'#fca5a5');assert.match(h.node('dq-table-tbody').innerHTML,/时间切分[\s\S]*❌/);
 h=setup();const missing=h.node('btn-dq-check').click();h.requests[0].resolve({ok:true,json:async()=>({...data,reproducibility:{}})});await missing;assert.match(h.node('dq-res').textContent,/未通过/);
 h=setup();const failed=h.node('btn-o11y-trace').click();h.requests[0].reject(new Error('fixture offline'));await failed;assert.match(h.node('o11y-trace').textContent,/证据加载失败/);
 for(const id of ['o11y-job','o11y-asset','dq-dataset','dq-step','dq-unit','dq-th','rbac-user'])assert.match(html,new RegExp(`<input[^>]+id="${id}"[^>]+readonly`));
 assert.match(html,/btn-o11y-trace" title="GET \/api\/v3\/mlops\/evidence/);assert.match(html,/btn-rca-run"[^>]*>查看准入拦截项/);
 console.log('MLOPS_EVIDENCE_CONTRACT:PASS:5 (delayed DOM and readonly scope fixtures; no browser claim)');
})().catch(error=>{console.error(error);process.exitCode=1;});
