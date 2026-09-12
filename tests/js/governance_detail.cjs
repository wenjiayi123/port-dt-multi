const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').resolve(__dirname, '../../app/ui/index.html'), 'utf8');
const start = source.indexOf('  let governance=null,');
const end = source.indexOf("  byId('btn-rbac-who')", start);
let code = source.slice(start, end);
const renderStart = code.indexOf('  function render(j){');
const renderEnd = code.indexOf('  function governanceError', renderStart);
// Keep real loading/selection handlers; replace table painting with a tiny DOM sink.
code = code.slice(0, renderStart) + 'function render(j){governance=j;renderGovernanceDetail();}\n' + code.slice(renderEnd);
const elements = new Map();
function byId(id){
  if(!elements.has(id)) elements.set(id, {textContent:'', handlers:{}, addEventListener(type,fn){this.handlers[type]=fn;}});
  return elements.get(id);
}
const requests=[];
const ctx={byId,console,fetch:()=>new Promise(resolve=>requests.push(resolve))};
vm.createContext(ctx);vm.runInContext(code+'\nthis.loadEvidence=load;',ctx);
const settle=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
(async()=>{
  const first=ctx.loadEvidence();
  const controls=byId('btn-gov-controls').handlers.click();
  const site=byId('btn-gov-site').handlers.click();
  assert.equal(requests.length,1,'startup and clicks must share one pending request');
  requests.shift()({ok:true,json:async()=>({controls:['old'],release_gate:{decision:'BLOCK'},boundary:{production_authority:false}})});
  await Promise.all([first,controls,site]);
  assert.equal(JSON.parse(byId('gov-detail').textContent).release_gate.decision,'BLOCK','latest selected detail wins');
  const refresh=ctx.loadEvidence();
  requests.shift()({ok:true,json:async()=>({controls:['new'],release_gate:{decision:'REVIEW'},boundary:{production_authority:false}})});
  await refresh;
  assert.equal(JSON.parse(byId('gov-detail').textContent).release_gate.decision,'REVIEW','refresh updates, never resets, selected view');
  await byId('btn-gov-controls').handlers.click();
  assert.deepEqual(JSON.parse(byId('gov-detail').textContent).controls,['new']);
  const failure=byId('btn-gov-refresh').handlers.click();
  requests.shift()({ok:false,status:503});await failure;await settle();
  assert.match(byId('gov-status').textContent,/503/,'load failure is visible');
  assert.deepEqual(JSON.parse(byId('gov-detail').textContent).controls,['new'],'failure preserves last evidence alongside the error');
  console.log('GOVERNANCE_DETAIL:PASS:coalescing,last-click,refresh,error');
})().catch(error=>{console.error(error);process.exitCode=1;});
