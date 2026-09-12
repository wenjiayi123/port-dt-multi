const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const test=require('node:test');
const copilot=fs.readFileSync('app/ui/ops_copilot.html','utf8');
const hub=fs.readFileSync('app/ui/integration_hub.html','utf8');
function source(html,name){const pattern=new RegExp('    (?:async )?function '+name+'\\(');const start=html.search(pattern);assert(start>=0,name);const tail=html.slice(start+1);const next=tail.search(/\n    (?:async )?function /);return html.slice(start,next<0?undefined:start+1+next);}
function node(){return {focus(){},showModal(){this.open=true},close(){this.open=false},textContent:'',value:'',disabled:false,style:{},classList:{add(){},remove(){}},reportValidity(){return true}};}
function base(extra={}){const nodes={};return vm.createContext({URL,Promise,Number,Boolean,String,Set,...extra,$:selector=>nodes[selector]||(nodes[selector]=node()),nodes});}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
for(const cancelDelay of [1100,650])test(`reset cancels pending RL navigation during ${cancelDelay}ms handoff delay`,async()=>{
 let resume;const routes=[];const progress=[];
 const c=base({answerRunId:1,generationCount:1,THINKING_DELAY_MS:2000,
  postJSON:async()=>({action:{id:'start_rl_training'}}),setProgress:(n)=>progress.push(n),
  wait:ms=>ms===cancelDelay?new Promise(resolve=>resume=resolve):Promise.resolve(),
  renderRlTrainingPlan(){},rlTrainingAnswer:()=>'',typeText:async()=>true,guidedHomeUrl:()=>'/rl-panel',
  window:{location:{assign:x=>routes.push(x)}}});
 vm.runInContext(source(copilot,'handoffRlTrainingCommand'),c);
 const pending=c.handoffRlTrainingCommand({query:'train'},1);await tick();assert(resume);c.answerRunId=2;resume();await pending;
 assert.deepEqual(routes,[]);if(cancelDelay===1100)assert(!progress.includes(92));
});
test('a request rejected after reset cannot overwrite the reset page',async()=>{
 let reject;const c=base({answerRunId:1,generationCount:0,THINKING_TEXT:'thinking',RETRIEVING_TEXT:'retrieving',THINKING_DELAY_MS:0,stages:['context','answer'],
  payload:()=>({query:'status',engine:'local_rag'}),clearAnswerState(){},setProgress(){},wait:async()=>{},
  isRlTrainingCommand:()=>false,postJSON:()=>new Promise((_r,j)=>reject=j),
  window:{setInterval:()=>1,clearInterval(){}},renderBrief:async()=>true});
 vm.runInContext(source(copilot,'ask'),c);const pending=c.ask();await tick();assert(reject);
 c.answerRunId+=1;c.$('#headline').textContent='reset state';reject(new Error('old failure'));await pending;
 assert.equal(c.$('#headline').textContent,'reset state');
});
test('confirmed route action follows the gateway URL in the same origin',async()=>{
 const routes=[];let health=0;
 const c=base({pendingPacket:{action:{id:'open_ops_copilot'}},clickMappedButton:async()=>false,
  buildAssistantPayload:()=>({}),api:async()=>({action:{label:'copilot'},execution_result:{executed:true,status:'ready',result:{type:'open_route',url:'http://localhost:8099/ops-copilot?mission=handoff'}}}),
  setText(){},pretty:JSON.stringify,typeAnswer:async()=>{},addLog(){},refreshHealth:async()=>health++,window:{location:{origin:'http://localhost:8099',assign:x=>routes.push(x)}}});
 vm.runInContext(source(hub,'confirmAssistantAction'),c);await c.confirmAssistantAction();
 assert.deepEqual(routes,['/ops-copilot?mission=handoff']);assert.equal(health,0);
});
test('unavailable desktop action reports its boundary instead of claiming a click',async()=>{
 let answer='';const c=base({document:{querySelector:()=>({disabled:true,title:'desktop integration disabled'})},typeAnswer:async x=>answer=x,addLog(){}});
 vm.runInContext(source(hub,'clickMappedButton'),c);
 assert.equal(await c.clickMappedButton({action:{id:'open_sailing_simulator'},will_execute:{button:{selector:'#launch'}}}),true);
 assert(answer.includes('没有启动进程'));
});
test('invalid numeric training parameters are not applied',()=>{
 const statuses=[];let read=false;const c=base({document:{querySelectorAll:()=>[{type:'number',value:'-1',setCustomValidity(){},reportValidity:()=>false}]},
  setText:(_s,t)=>statuses.push(t),trainingConfigFromUi:()=>{read=true;return {}}});
 vm.runInContext(source(hub,'applyAdvancedParamsAndReturn'),c);c.applyAdvancedParamsAndReturn();
 assert.equal(read,false);assert(statuses.includes('参数无效 · 未应用'));
});
test('unavailable context never appears as connected or hash-verified',()=>{
 const c=base({currentContext:null,escapeHTML:String,levelClass:()=>'',renderMissionRail(){}});
 vm.runInContext(source(copilot,'renderContext'),c);c.renderContext({status:'unavailable',port:'CNSHA',signals:[],connectors:[]});
 assert(c.$('#liveHeadline').textContent.includes('上下文不可用'));
 assert(!c.$('#liveHeadline').textContent.includes('已接入'));
  assert(c.$('#liveSub').textContent.includes('不能视为现场已接入'));
});
test('handoff metadata changed while a request is pending keeps confirmation disabled',()=>{
 const c=base({lastHandoff:null});
 c.$('#mission').value='handoff';c.$('#operator').value='new operator';c.$('#shift').value='QA';
 vm.runInContext(source(copilot,'handoffCanConfirm'),c);vm.runInContext(source(copilot,'updateHandoffPanel'),c);
 c.updateHandoffPanel({operator:'previous operator',shift:'QA',context_sha256:'abc',handoff_sha256:'def'});
 assert.equal(c.handoffCanConfirm(),false);assert.equal(c.$('#confirmHandoff').disabled,true);
 assert(c.$('#handoffMeta').textContent.includes('请刷新预览'));
});
test('handoff and triage output selectors synchronize the visible mission',()=>{
 for(const [mode,mission] of [['handoff','handoff'],['alert_triage','triage']]){
  const selected=[];const c=base({selectMission:(value)=>selected.push(value),invalidateAnswer(){}});
  c.$('#mission').value='situation';c.$('#mode').value=mode;
  vm.runInContext(source(copilot,'selectOutputMode'),c);c.selectOutputMode();
  assert.deepEqual(selected,[mission]);assert.equal(c.$('#mode').value,mode);
 }
});

test('current request failure clears pending answer, audit and old context proof',async()=>{
 const c=base({answerRunId:0,contextRunId:3,generationCount:0,lastAudit:{old:true},lastHandoff:{old:true},currentContext:{status:'review'},
  THINKING_TEXT:'thinking',RETRIEVING_TEXT:'retrieving',THINKING_DELAY_MS:0,stages:['context','answer'],
  payload:()=>({query:'status',engine:'local_rag',port:'SGSIN'}),clearAnswerState(){},setProgress(){},wait:async()=>{},
  isRlTrainingCommand:()=>false,postJSON:async()=>{throw new Error('422 unsupported_port')},
  window:{setInterval:()=>1,clearInterval(){}}});
 vm.runInContext(source(copilot,'ask'),c);await c.ask();
 assert(c.$('#operatorNote').textContent.includes('未生成可用答案'));
 assert(c.$('#auditBox').textContent.includes('未生成审计包'));
 assert.equal(c.$('#ctxStatus').textContent,'Context INVALID');assert.equal(c.currentContext,null);
 for(const selector of ['#sourceRail','#policyRail','#gateRail','#hashRail'])assert.equal(c.$(selector).textContent,'本次上下文无效');
 assert.equal(c.lastAudit,null);assert.equal(c.lastHandoff,null);assert.equal(c.$('#askBtn').disabled,false);
 assert.equal(c.contextRunId,4);assert.equal(c.$('#copyAudit').disabled,true);
});
function handoffHarness(){
 const posts=[];const c=base({answerRunId:3,contextRunId:2,handoffConfirmation:null,
  lastHandoff:{operator:'QA',shift:'day',context_sha256:'context-sha',handoff_sha256:'handoff-sha'},
  currentContext:{context_sha256:'context-sha'},persistHandoff:async confirmed=>posts.push(confirmed)});
 c.$('#mission').value='handoff';c.$('#operator').value='QA';c.$('#shift').value='day';c.$('#port').value='CNSHA';c.$('#asset').value='qc-01';
 for(const name of ['handoffCanConfirm','handoffConfirmationKey','openHandoffConfirmation','cancelHandoffConfirmation','approveHandoffConfirmation'])vm.runInContext(source(copilot,name),c);
 return {c,posts};
}
test('handoff dialog requires separate review and confirm; cancel posts nothing',async()=>{
 const {c,posts}=handoffHarness();c.openHandoffConfirmation();
 assert.equal(c.$('#handoffConfirmDialog').open,true);assert.deepEqual(posts,[]);
 for(const detail of ['QA','day','context-sha','handoff-sha','生产动作=false'])assert(c.$('#handoffConfirmDetails').textContent.includes(detail));
 c.cancelHandoffConfirmation();await c.approveHandoffConfirmation();assert.deepEqual(posts,[]);
 c.openHandoffConfirmation();let prevented=false;
 const cancelHandler=copilot.match(/"cancel", (event=>\{event\.preventDefault\(\);cancelHandoffConfirmation\(\);\})\);/)[1];
 vm.runInContext('globalThis.escapeCancel='+cancelHandler,c);c.escapeCancel({preventDefault(){prevented=true}});
 assert.equal(prevented,true);assert.equal(c.$('#handoffConfirmDialog').open,false);assert.deepEqual(posts,[]);
 c.openHandoffConfirmation();await c.approveHandoffConfirmation();await c.approveHandoffConfirmation();
 assert.deepEqual(posts,[true]);assert.equal(c.$('#handoffConfirmDialog').open,false);
 assert(!copilot.includes('window.confirm('));
 assert(copilot.includes('"cancel", event=>{event.preventDefault();cancelHandoffConfirmation();}'));
});
test('handoff approval rejects a changed context, preview, or operator without posting',async()=>{
 for(const change of [c=>c.contextRunId++,c=>c.lastHandoff.handoff_sha256='changed',c=>c.$('#operator').value='other',c=>c.answerRunId++]){
  const {c,posts}=handoffHarness();c.openHandoffConfirmation();change(c);await c.approveHandoffConfirmation();
  assert.deepEqual(posts,[]);assert(c.$('#handoffMeta').textContent.includes('本次未留痕'));
 }
});
