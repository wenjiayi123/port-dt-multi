// Execute the actual home-rule controller with DOM fixtures; no browser claim.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.resolve(__dirname,'../../app/ui/index.html'),'utf8');
const controller=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].find(m=>m[1].includes('function buildHomeRules()'))[1];
function fixture(status,peak,pending){
  const nodes=new Map();
  const node=id=>{if(!nodes.has(id))nodes.set(id,{textContent:''});return nodes.get(id);};
  node('exec-status').textContent=status;node('exec-peak').textContent=peak;node('ap-pending').textContent=pending;
  const context={document:{getElementById:node},window:{addEventListener(){}},setTimeout(){},console};
  vm.createContext(context);vm.runInContext(controller,context);
  const sync=()=>context.window.__syncHomeHero();sync();return {node,context,sync};
}
let checks=0;
for(const [status,peak,pending] of [
  ['待接入港口','待接入港口','待接入港口'],
  ['稳定','待接入港口','待接入港口'],
  ['稳定','0 %','待接入港口'],
  ['稳定','待接入港口','0'],
  ['证据受限','0 %','0'],
  ['未验证','0 %','0'],
  ['稳定','待接入：0%','待接入：0'],
  ['稳定','',''],
]){
  const f=fixture(status,peak,pending);
  assert.equal(f.node('home-status').textContent,'证据待补');
  assert.equal(f.node('home-risk').textContent,'风险未评定');
  assert.doesNotMatch(f.node('home-risk-sub').textContent,/当前无待审批/);
  if(pending!=='0'){
    assert.match(f.node('home-loop-dispatch').textContent,/待办未接入/);
    assert.doesNotMatch(f.node('home-box-2').textContent,/待审批=0/);
  }
  checks++;
}
for(const status of ['稳定','healthy']){
  const f=fixture(status,'0 %','0');assert.equal(f.node('home-status').textContent,'稳定');assert.equal(f.node('home-risk').textContent,'中低');assert.match(f.node('home-risk-sub').textContent,/当前无待审批工单/);assert.match(f.node('home-loop-dispatch').textContent,/待审批=0/);checks++;
}
for(const [status,peak,pending,expected] of [['证据受限','30 %','待接入港口','高'],['待接入港口','待接入港口','5','高'],['critical','待接入港口','待接入港口','高'],['稳定','10 %','待接入港口','中高']]){
  const f=fixture(status,peak,pending);assert.equal(f.node('home-risk').textContent,expected);assert.match(f.node('home-risk-sub').textContent,/尚缺.*证据/);checks++;
}
{
  const f=fixture('稳定','0 %','0');f.node('ap-pending').textContent='待接入港口';f.sync();assert.equal(f.node('home-risk').textContent,'风险未评定');f.node('ap-pending').textContent='0';f.sync();assert.equal(f.node('home-risk').textContent,'中低');checks++;
}
{
  const f=fixture('稳定','0 %','0');f.node('qc-pending-count').textContent='待接入港口';f.sync();assert.equal(f.node('home-risk').textContent,'风险未评定','current visible approval source overrides old legacy placeholder');f.node('qc-pending-count').textContent='0';f.sync();assert.equal(f.node('home-risk').textContent,'中低');checks++;
}
const sidebar=html.slice(html.indexOf('    const approvalPending=realtimeEvidence?.approvals'),html.indexOf("    if($('#qc-last-job'))",html.indexOf('    const approvalPending=realtimeEvidence?.approvals')));
assert.ok(sidebar.includes('qc-pending-count'));
for(const [available,pending,expected] of [[true,null,'待接入港口'],[true,undefined,'待接入港口'],[true,false,'待接入港口'],[false,0,'待接入港口'],[true,0,'0'],[true,'0','0']]){
  const output={textContent:''},context={$:()=>output,realtimeEvidence:{approvals:{available,pending}}};vm.createContext(context);vm.runInContext(sidebar,context);assert.equal(output.textContent,expected);checks++;
}
console.log(`HOME_EVIDENCE_RULES_CONTRACT:PASS:${checks} (actual controller, offline DOM fixtures; no browser claim)`);
