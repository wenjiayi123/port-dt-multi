// DOM contract unit tests. No browser, live HTTP requests or model training.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');

function harness(script, html = '') {
  const nodes = new Map();
  const groups = new Map();
  class Element {
    constructor(id) { this.id=id; this.value=''; this.textContent=''; this.innerHTML=''; this.style={}; this.disabled=false; this.handlers={}; this.attrs={}; this.children=[]; this.options=[]; const classes=new Set(); this.classList={add:(...items)=>items.forEach(x=>classes.add(x)),remove:(...items)=>items.forEach(x=>classes.delete(x)),contains:x=>classes.has(x),toggle:(x,on)=>on?classes.add(x):classes.delete(x)}; }
    get selectedOptions(){ return this.options.filter(x=>x.value===this.value); }
    addEventListener(type, fn){ this.handlers[type]=fn; }
    setAttribute(name, value){ this.attrs[name]=value; }
    getAttribute(name){ return this.attrs[name]; }
    setCustomValidity(message){ this.validationMessage=message; }
    reportValidity(){ if(this.disabled||this.readOnly)return true; const n=Number(this.value),step=Number(this.attrs.step||1),base=Number(this.attrs.min??this.attrs.value??0);const stepValid=this.attrs.step==='any'||Math.abs((n-base)/step-Math.round((n-base)/step))<1e-7;return !this.validationMessage && (this.attrs.type!=='number' || (Number.isFinite(n) && (this.attrs.min===undefined || n>=Number(this.attrs.min)) && (this.attrs.max===undefined || n<=Number(this.attrs.max))&&stepValid)); }
    focus(){ document.activeElement=this; }
    click(){ return this.handlers.click?.({target:this,preventDefault(){}}); }
    appendChild(child){ this.children.push(child); }
    remove(){}
    querySelector(selector){ return selector==='em' ? node(this.id+'-label') : null; }
    scrollIntoView(){}
    getContext(){ return new Proxy({}, {get:()=>()=>{}}); }
    get clientWidth(){ return 640; }
    get clientHeight(){ return 240; }
  }
  const node=id=>{ if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id); };
  for(const match of html.matchAll(/<(input|select)\b([^>]*)>/g)){
    const attrs=Object.fromEntries([...match[2].matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
    if(!attrs.id)continue;
    const el=node(attrs.id);el.attrs=attrs;el.value=attrs.value||'';
    if(match[1]==='select'){
      const tail=html.slice(match.index+match[0].length).split('</select>')[0];
      el.options=[...tail.matchAll(/<option\b[^>]*value="([^"]*)"[^>]*>([^<]*)/g)].map(m=>({value:m[1],textContent:m[2]}));
      el.value=el.options[0]?.value||'';
    }
  }
  groups.set('.train-card input',[...nodes.values()].filter(x=>x.attrs.type==='number' && x.id!=='inpEvalEpisodes'));
  groups.set("input[name='pick']",[]);
  const document={body:node('body'),activeElement:null,hidden:false,
    getElementById:node,createElement:tag=>new Element(tag),addEventListener(){},
    querySelector(selector){ if(selector.startsWith('#'))return node(selector.slice(1)); if(selector==='.run-stage.active')return [...nodes.values()].find(x=>x.classList.contains('active'))||null;return null; },
    querySelectorAll:selector=>groups.get(selector)||[]};
  const context={document,console,URL,URLSearchParams,Intl,Date,Math,Number,JSON,Promise,Array,Object,Set,Map,AbortController,
    setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},devicePixelRatio:1,
    fetch:async()=>({ok:true,json:async()=>({}),text:async()=>''})};
  context.window={location:{search:'',hash:'',origin:'http://unit.test',assign(){}},addEventListener(){},setTimeout:context.setTimeout,setInterval:context.setInterval,clearInterval(){},alert(){},history:{length:1}};
  vm.createContext(context);vm.runInContext(script,context);
  return {context,node,groups,run:code=>vm.runInContext(code,context)};
}

const server=fs.readFileSync(path.join(root,'app/server.py'),'utf8');
const panelHTML=server.split('_RL_PANEL_HTML = r"""')[1].split('"""')[0];
const panelScript=panelHTML.match(/<script>([\s\S]*?)<\/script>/)[1];
function panel(){ const h=harness(panelScript,panelHTML);h.node('selDataset').value='fixture_dataset';h.run("trainingDatasetCatalog=[{dataset_id:'fixture_dataset'}]");return h; }

(async()=>{
  let checks=0;
  {
    const h=panel();h.run("requestTrainStart=async cfg=>{globalThis.submitted=cfg;return {job_id:'fixture-job',status:'QUEUED'}};renderTrainingStatus=()=>{};startStatusPolling=()=>{}");
    h.run('showAssistantRunConfirm()');
    const reviewed=JSON.parse(h.node('confirmConfigSnapshot').textContent);
    assert.equal(reviewed.dataset_id,'fixture_dataset');assert.equal(reviewed.seed,42);assert.equal(reviewed.replay_buffer,120000);assert.equal(reviewed.tau,.005);
    h.node('inpSeed').value='777';h.node('inpTotalSteps').value='640';
    await h.node('btnConfirmAssistantRun').click();
    assert.equal(h.context.submitted.seed,42);assert.equal(h.context.submitted.total_steps,20000,'submit the reviewed snapshot, not changed background fields');
    h.context.submitted=undefined;
    h.run('lastTrainStatus=null;showAssistantRunConfirm()');await h.node('btnCancelAssistantRun').click();await h.node('btnConfirmAssistantRun').click();
    assert.equal(h.context.submitted,undefined,'cancelled configuration cannot later be submitted');checks++;
  }
  {
    const h=panel();h.node('inpTotalSteps').value='';h.run('showAssistantRunConfirm()');assert.notEqual(h.node('assistantConfirmBackdrop').style.display,'flex');checks++;
  }
  {
    for(const file of fs.readdirSync(path.join(root,'config/ports')).filter(x=>x.endsWith('.json'))){
      const profile=JSON.parse(fs.readFileSync(path.join(root,'config/ports',file),'utf8'));
      if(!profile.assets?.demand_cap_kw)continue;
      const h=panel();h.node('inpDemandCap').value=String(profile.assets.demand_cap_kw);h.node('inpTotalSteps').value='64';h.node('inpSeed').value='912';
      for(const [id,key] of [['inpCostW','cost'],['inpCarbonW','carbon'],['inpPeakW','peak'],['inpSafetyW','safety']])if(profile.objectives?.[key]!==undefined)h.node(id).value=String(profile.objectives[key]);
      h.run('showAssistantRunConfirm()');assert.equal(h.node('assistantConfirmBackdrop').style.display,'flex',`${file}: profile defaults and 64-step budget must pass actual min/max/step constraints`);
    }checks++;
  }
  {
    const h=panel();assert.equal(h.run("isTrainableDataset({quality:{training_eligible:true},split_policy:{role:'forward_challenge_only',candidate_selection_allowed:false}})"),false);
    h.run("trainingDatasetCatalog=[{dataset_id:'fixture_dataset',split_policy:{candidate_selection_allowed:false}}];showAssistantRunConfirm()");assert.notEqual(h.node('assistantConfirmBackdrop').style.display,'flex');checks++;
  }
  {
    const h=panel();h.run("trainJobId='keep-this-job';lastTrainStatus={status:'RUNNING'};fetch=async()=>({ok:false,text:async()=> '503 temporary failure'})");
    await h.run('resetTraining()');assert.equal(h.run('trainJobId'),'keep-this-job');assert.equal(h.run('lastTrainStatus.status'),'RUNNING');assert.match(h.node('trainDetail').textContent,/取消失败/);checks++;
  }
  {
    const h=panel();h.run("trainJobId='await-cancel';lastTrainStatus={status:'RUNNING'};fetch=async()=>({ok:true,json:async()=>({status:'RUNNING',stage:'cancelling'})});renderTrainingStatus=()=>{};startStatusPolling=()=>{}");
    await h.run('resetTraining()');assert.equal(h.run('trainJobId'),'await-cancel');assert.match(h.node('trainDetail').textContent,/后端确认停止/);checks++;
  }
  {
    const h=panel();h.run("trainJobId='race-job';lastTrainStatus={job_id:'race-job',status:'RUNNING'};fetch=async()=>({ok:false,text:async()=> 'already completed'});pollTrainingStatus=async()=>renderTrainingStatus({job_id:'race-job',status:'COMPLETED',evaluation_available:true},'fixture')");
    await h.run('pauseTraining()');assert.equal(h.node('btnPauseTrain').disabled,true);assert.equal(h.node('btnPauseTrain').textContent,'暂停');assert.match(h.node('trainDetail').textContent,/训练已完成/);
    h.run("renderTrainingStatus({job_id:'race-job',status:'PAUSED'},'late response')");assert.equal(h.run('lastTrainStatus.status'),'COMPLETED');assert.equal(h.node('btnPauseTrain').disabled,true);checks++;
  }
  {
    const h=panel();h.node('inpEvalEpisodes').value='7';h.run("selectedId='fixture-model';fetch=async(url,options)=>{globalThis.simPayload=JSON.parse(options.body);return {ok:true,json:async()=>({summary:{delta_kWh:1,delta_carbon_kg:null,window:{}},baseline:{agg_kW:[2,3]},simulated:{agg_kW:[1,2]}})}}");
    await h.run('simulate()');assert.deepEqual(JSON.parse(JSON.stringify(h.context.simPayload)),{strategy_id:'fixture-model',episodes:7});assert.equal(h.node('m_dco2').textContent,'N/A');
    h.run("fetch=async()=>({ok:false,text:async()=> 'failed'})");await h.run('simulate()');assert.equal(h.node('m_dkwh').textContent,'—');assert.equal(h.run('lastSimulation'),null);checks++;
  }
  {
    const h=panel();h.run("trainJobId='eval-job';lastTrainStatus={job_id:'eval-job',status:'COMPLETED',evaluation_available:true};globalThis.evaluationCalls=0;fetch=async()=>{evaluationCalls++;return await new Promise(resolve=>globalThis.finishEvaluation=resolve)};pollTrainingStatus=async()=>renderTrainingStatus(lastTrainStatus,'late poll')");
    const pending=h.run('evaluateTraining()');await h.run('evaluateTraining()');assert.equal(h.context.evaluationCalls,1);assert.equal(h.node('btnStartTrain').disabled,true);assert.equal(h.node('btnResetTrain').disabled,true);assert.match(h.node('trainDetail').textContent,/正在对.*10 回合/);
    h.context.finishEvaluation({ok:true,json:async()=>({evaluation_id:'ui-eval-fixture',formal_evidence_updated:false,model_registry_updated:false,algorithm:'sac',metrics:{reward:1,carbon_kg:null},render:{frames:[],frame_count:0}})});await pending;
    assert.match(h.node('trainDetail').textContent,/已完成并保存：ui-eval-fixture/);assert.equal(h.node('trainStatus').textContent,'COMPLETED');assert.match(h.node('evaluationMetrics').textContent,/carbon=N\/A/);assert.equal(h.node('btnEvaluateTrain').disabled,false);assert.equal(h.node('btnResetTrain').disabled,false);
    h.run("renderTrainingStatus(lastTrainStatus,'manual poll')");assert.match(h.node('trainDetail').textContent,/已完成并保存/);checks++;
  }
  {
    const h=panel();h.run("trainJobId='eval-job';lastTrainStatus={job_id:'eval-job',status:'COMPLETED',evaluation_available:true};fetch=async()=>({ok:false,text:async()=> 'fixture failure'})");await h.run('evaluateTraining()');assert.match(h.node('trainDetail').textContent,/独立交互评测失败.*fixture failure/);assert.equal(h.node('btnEvaluateTrain').disabled,false);assert.match(h.node('trainStage').textContent,/Evaluation failed/);assert.equal(h.run('lastTrainStatus.status'),'COMPLETED');checks++;
  }
  {
    const h=panel();h.run("renderTrainingStatus({job_id:'monotone-job',status:'RUNNING',updated_at:'2026-09-12T04:01:00Z'});renderTrainingStatus({job_id:'monotone-job',status:'PAUSED',updated_at:'2026-09-12T04:00:00Z'})");assert.equal(h.run('lastTrainStatus.status'),'RUNNING');assert.equal(h.node('btnPauseTrain').textContent,'暂停');
    h.run("renderTrainingStatus({job_id:'monotone-job',status:'PAUSED',updated_at:'2026-09-12T04:02:00Z'})");assert.equal(h.node('btnPauseTrain').textContent,'继续');h.run("renderTrainingStatus({job_id:'monotone-job',status:'CANCELLED',updated_at:'2026-09-12T04:03:00Z'});renderTrainingStatus({job_id:'monotone-job',status:'RUNNING',updated_at:'2026-09-12T04:04:00Z'})");assert.equal(h.run('lastTrainStatus.status'),'CANCELLED');assert.equal(h.node('btnPauseTrain').disabled,true);assert.match(h.node('trainDetail').textContent,/已确认取消/);checks++;
  }
  {
    const h=panel();h.run("trainJobId='eval-job';lastTrainStatus={job_id:'eval-job',status:'COMPLETED',evaluation_available:true};fetch=async()=>({ok:true,json:async()=>({algorithm:'sac',metrics:{reward:7}})})");await h.run('evaluateTraining()');assert.match(h.node('trainDetail').textContent,/未收到完整独立评测保存回执/);assert.equal(h.run('trainingEvaluation.state'),'FAILED');assert.doesNotMatch(h.node('evaluationMetrics').textContent,/reward=7/);assert.equal(h.node('btnEvaluateTrain').disabled,false);checks++;
  }
  {
    const h=panel();h.run("selectedId='old-model';currentList=[{id:'old-model'}];fetch=async()=>({ok:false,text:async()=> '503'})");await h.run('loadList()');assert.equal(h.run('selectedId'),null);assert.equal(h.node('btnSim').disabled,true);assert.equal(h.node('btnVerifyDryRun').disabled,true);checks++;
  }
  {
    const h=panel();h.run("globalThis.extraStarts=0;globalThis.pollStarts=0;startTraining=async()=>extraStarts++;startStatusPolling=()=>pollStarts++;renderTrainingStatus=()=>{};loadMobileTrainingRequests=async()=>{};fetch=async()=>({ok:true,json:async()=>({job_id:'already-created',config:{},training_status:{status:'QUEUED'}})})");await h.run("reviewMobileTrainingRequest('fixture-request','approve')");assert.equal(h.context.extraStarts,0);assert.equal(h.context.pollStarts,1);checks++;
  }
  {
    const h=panel();assert.match(h.run('fmtImpact({reward:null,peak_kw:null,guardrail_violation_rate:null})'),/reward:N\/A.*violations:N\/A/);checks++;
  }
  {
    const h=panel();h.node('selAlgo').value='sac';h.run('updateConnectorPreview()');assert.equal(h.node('inpEntropy').disabled,true);assert.equal(h.run('trainConfig().entropy_coef'),undefined);assert.match(h.node('entropyContractHint').textContent,/自动熵/);
    h.node('selAlgo').value='ppo';h.run('updateConnectorPreview()');assert.equal(h.node('inpEntropy').disabled,false);assert.equal(h.run('trainConfig().entropy_coef'),.02);
    h.node('selAlgo').value='trpo';h.run('updateConnectorPreview()');assert.equal(h.node('inpEntropy').disabled,true);assert.equal(h.run('trainConfig().entropy_coef'),undefined);assert.match(h.node('entropyContractHint').textContent,/不使用/);checks++;
  }
  {
    const h=panel();h.run("trainingDatasetCatalog=[{dataset_id:'dataset-a',port_profile_id:'a'},{dataset_id:'dataset-b',port_profile_id:'b'}];globalThis.profileResolvers={};updateConnectorPreview=()=>{};updateBaselineMetrics=async()=>{};fetch=url=>new Promise(resolve=>{profileResolvers[url.split('/').pop()]=resolve})");
    h.node('selDataset').value='dataset-a';const first=h.run('syncSelectedDatasetContract()');
    h.node('selDataset').value='dataset-b';const second=h.run('syncSelectedDatasetContract()');
    h.context.profileResolvers.b({ok:true,json:async()=>({objectives:{carbon:16}})});await second;
    h.context.profileResolvers.a({ok:true,json:async()=>({objectives:{carbon:12}})});await first;
    assert.equal(h.node('inpCarbonW').value,'16','late profile from previous dataset cannot replace current dataset weights');checks++;
  }
  {
    const h=panel();h.run("globalThis.baselineResolvers=[];fetch=()=>new Promise(resolve=>baselineResolvers.push(resolve))");
    const first=h.run('updateBaselineMetrics()');const second=h.run('updateBaselineMetrics()');
    h.context.baselineResolvers[1]({ok:true,json:async()=>({baselines:[{id:'sac',status:'READY',latest_evaluation:{metrics:{reward:7,carbon_kg:null,peak_kw:0}}}]})});await second;
    h.context.baselineResolvers[0]({ok:true,json:async()=>({baselines:[{id:'sac',status:'OLD',latest_evaluation:{metrics:{reward:1}}}]})});await first;
    assert.equal(h.node('algo_sac_reward').textContent,'7.0000');assert.equal(h.node('algo_sac_carbon').textContent,'—');assert.equal(h.node('algo_sac_peak').textContent,'0.0');assert.equal(h.node('algo_sac_status').textContent,'READY');checks++;
  }
  {
    const h=harness(fs.readFileSync(path.join(root,'app/ui/v3/v3.js'),'utf8'));
    for(const missing of ['null','undefined',"''"]){assert.equal(h.run(`safeNumber(${missing})`),'—');assert.equal(h.run(`pct(${missing})`),'—');assert.equal(h.run(`metricValue('carbon_kg',${missing})`),'—');}
    assert.equal(h.run('safeNumber(0)'),'0.0');assert.equal(h.run("liveValue({wave:null},'wave')"),null);checks++;
  }
  {
    const h=harness(fs.readFileSync(path.join(root,'app/ui/rl_future/rl_future.js'),'utf8'));
    assert.throws(()=>h.run('validateRunPayload({})'),/回执不完整/);
    h.node('riskValue').textContent='DRY-RUN';h.node('stabilityValue').textContent='100%';h.node('trustValue').textContent='95% CI';
    h.run("fetch=async()=>({ok:false,json:async()=>({detail:'fixture failure'})})");await h.run('ignite()');
    assert.equal(h.node('riskValue').textContent,'BLOCKED');assert.equal(h.node('stabilityValue').textContent,'--');assert.equal(h.node('trustValue').textContent,'--');assert.match(h.node('aiSummary').textContent,/本次推演失败/);assert.equal(h.node('btnIgnite').disabled,false);checks++;
  }
  console.log(`RL_PAGES_DOM_CONTRACT:PASS:${checks} (unit fixtures; no real browser coverage claimed)`);
})().catch(error=>{console.error(error);process.exitCode=1;});
