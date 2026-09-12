// Execute actual Integration Hub functions with HTML-derived controls.
// Numeric validity reproduces HTML min/max/step semantics; this is not a browser run.
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm'), test=require('node:test');
const html=fs.readFileSync('app/ui/integration_hub.html','utf8');
// Exact boundaries of the trusted inline controller; do not treat this as a sanitizer.
const scriptStart=html.indexOf('<script>');
assert(scriptStart>=0,'integration inline script opening marker must exist');
const scriptEnd=html.indexOf('</script>',scriptStart+'<script>'.length);
assert(scriptEnd>scriptStart,'integration inline script closing marker must exist');
const initStart=html.indexOf('    init().catch',scriptStart+'<script>'.length);
assert(initStart>scriptStart&&initStart<scriptEnd,'integration bootstrap boundary must exist inside the script');
const script=html.slice(scriptStart+'<script>'.length,initStart);
function harness(){
 const nodes=new Map(), controls=[];
 function element(id,attrs={}){
  const classes=new Set();return {id,attrs,type:attrs.type,value:attrs.value||'',disabled:false,textContent:'',innerHTML:'',title:'',dataset:{},style:{setProperty(){}},
   classList:{add:x=>classes.add(x),remove:(...xs)=>xs.forEach(x=>classes.delete(x)),toggle:(x,on)=>on?classes.add(x):classes.delete(x),contains:x=>classes.has(x)},
   setAttribute(k,v){this.attrs[k]=v},getAttribute(k){return this.attrs[k]},addEventListener(){},focus(){},closest(){return null},
   setCustomValidity(message){this.validationMessage=message},
   checkValidity(){
    if(this.disabled)return true;if(this.validationMessage)return false;
    if(this.type!=='number'&&this.type!=='range')return true;
    if(!this.value.trim())return !Object.hasOwn(this.attrs,'required');
    const n=Number(this.value),min=this.attrs.min===undefined?-Infinity:Number(this.attrs.min),max=this.attrs.max===undefined?Infinity:Number(this.attrs.max);
    if(!Number.isFinite(n)||n<min||n>max)return false;
    if(this.attrs.step==='any')return true;
    const step=Number(this.attrs.step||1),base=Number(this.attrs.min??this.attrs.value??0),ratio=(n-base)/step;
    return Math.abs(ratio-Math.round(ratio))<1e-7;
   },reportValidity(){return this.checkValidity()}};
 }
 function node(id){if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id)}
 for(const m of html.matchAll(/<(input|select)\b([^>]*)>/g)){
  const attrs=Object.fromEntries([...m[2].matchAll(/([\w-]+)="([^"]*)"/g)].map(a=>[a[1],a[2]]));if(!attrs.id)continue;
  const e=element(attrs.id,attrs);nodes.set(attrs.id,e);
  if(m[1]==='select'){
   const body=html.slice(m.index+m[0].length).split('</select>')[0],options=[...body.matchAll(/<option\b([^>]*)value="([^"]*)"([^>]*)>/g)];
   e.value=(options.find(x=>(x[1]+x[3]).includes('selected'))||options[0])?.[2]||'';
  }
  if(m.index>html.indexOf('id="trainingParamBackdrop"')&&m.index<html.indexOf('id="xiaoyiConfirmBackdrop"'))controls.push(e);
 }
 const document={body:node('body'),activeElement:null,querySelector:sel=>sel.startsWith('#')?node(sel.slice(1)):null,
  querySelectorAll:sel=>sel==='#trainingParamBackdrop input,#trainingParamBackdrop select'?controls:[],createElement:()=>element('created')};
 const c=vm.createContext({document,window:{location:{search:'',origin:'http://unit.test'},setTimeout(){}},URLSearchParams,URL,console,Set,Date,Number,JSON,Array,Object,Math,Promise});
 vm.runInContext(script,c);vm.runInContext('addLog=()=>{};',c);
 return {c,node,controls,run:code=>vm.runInContext(code,c)};
}
for(const target of ['yard_lighting','hvac_cooling','shore_bess','bess_energy','yard_crane'])test(`all numeric defaults valid; ${target} profile applies and returns`,()=>{
 const h=harness();h.run(`selectTrainingTarget('${target}',false)`);
 // Also inspect disabled archived numbers; none may ship with contradictory default limits.
 for(const control of h.controls){const disabled=control.disabled;control.disabled=false;assert(control.checkValidity(),`${control.id}: ${control.value}, ${JSON.stringify(control.attrs)}`);control.disabled=disabled;}
 h.run('openAdvancedParams()');assert(h.node('trainingParamBackdrop').classList.contains('open'));
 h.run('applyAdvancedParamsAndReturn()');assert(!h.node('trainingParamBackdrop').classList.contains('open'));
 assert(h.node('paramConfigState').textContent.includes('待训练面板复核'));
});
test('unsupported fields stay archived and cannot block Apply; actual controls track learner',()=>{
 const h=harness();h.run("selectTrainingTarget('shore_bess',false)");
 for(const id of ['trainOptimizer','trainLrSchedule','trainHiddenLayers','trainStepMin','trainGuardrail','trainSeed','sacAutoAlpha','trainMaxRamp','trainEntropyCoef'])assert(h.node(id).disabled,id);
 for(const id of ['trainSteps','trainLearningRate','trainHorizon','trainGamma','trainBatch','trainTau','trainReplayBuffer','weightCarbon'])assert(!h.node(id).disabled,id);
 h.node('trainMaxRamp').value='invalid-archived';h.run('openAdvancedParams();applyAdvancedParamsAndReturn()');assert(!h.node('trainingParamBackdrop').classList.contains('open'));
 h.run("selectTrainingTarget('yard_crane',false)");assert(!h.node('trainEntropyCoef').disabled);assert(h.node('trainTau').disabled);assert(h.node('trainReplayBuffer').disabled);
 h.node('trainSteps').value='20001';h.run('openAdvancedParams();applyAdvancedParamsAndReturn()');assert(h.node('trainingParamBackdrop').classList.contains('open'));assert.equal(h.node('paramConfigState').textContent,'参数无效 · 未应用');
});
test('global latest job never inherits the selected training profile without persisted binding',()=>{
 const h=harness();h.run("selectTrainingTarget('shore_bess',false)");
 h.c.fixture={job_id:'old-global-job',dataset_id:'old-dataset',algorithm:'sac',status:'EVALUATED',progress:100,manifest:{model_sha256:'abcdef0123456789'}};
 h.run('renderRlStatus({status:fixture})');assert(h.node('impactTargetState').textContent.includes('不能归属于本档案'));assert(!h.node('stageArtifact').classList.contains('done'));
 assert(h.node('impactTargetModel').textContent.includes('训练档案 shore_bess'));assert(!h.node('impactTargetModel').textContent.includes('abcdef'));
 assert(h.node('rlPolicy').textContent.includes('old-global-job'));assert(h.node('rlPolicy').textContent.includes('old-dataset'));
 h.c.fixture.config={module_target:'yard_crane'};h.run('renderRlStatus(fixture)');assert(!h.node('stageArtifact').classList.contains('done'));
 h.c.fixture.config.module_target='shore_bess';h.run('renderRlStatus(fixture)');assert(h.node('stageArtifact').classList.contains('done'));assert(h.node('impactTargetModel').textContent.includes('old-global-job'));
 h.run("selectTrainingTarget('yard_crane',false)");assert(!h.node('stageArtifact').classList.contains('done'));
});
test('failed latest-job refresh removes old completion and policy evidence',async()=>{
 const h=harness();h.run("selectTrainingTarget('shore_bess',false)");
 h.c.fixture={job_id:'previous-job',dataset_id:'fixture',algorithm:'sac',status:'EVALUATED',progress:100,config:{module_target:'shore_bess'}};
 h.run('renderRlStatus(fixture);api=async()=>{throw Error("HTTP 503")};');assert(h.node('stageArtifact').classList.contains('done'));
 await h.run('refreshRlStatus()');assert.equal(h.node('rlStatus').textContent,'UNAVAILABLE');assert(!h.node('stageArtifact').classList.contains('done'));
 assert(h.node('impactTargetState').textContent.includes('未核验'));assert(!h.node('rlPolicy').textContent.includes('previous-job'));
});
