const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const html=fs.readFileSync(require('node:path').resolve(__dirname,'../../app/ui/index.html'),'utf8');
const scripts=[...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)].map(x=>x[1]);
function section(start,end){const a=html.indexOf(start);assert(a>=0,start);const b=html.indexOf(end,a+start.length);assert(b>a,end);return html.slice(a,b);}
function element(){const handlers={};return {textContent:'',innerHTML:'',value:'',disabled:false,style:{},handlers,width:880,height:180,
 classList:{add(){},remove(){},toggle(){}},addEventListener(type,fn){(handlers[type]??=[]).push(fn)},async fire(type){for(const fn of handlers[type]||[])await fn({target:this})},
 querySelector(){return this.body??=element()},querySelectorAll(){return[]},replaceChildren(){this.innerHTML='';this.children=[]},appendChild(value){(this.children??=[]).push(value)},
 getBoundingClientRect(){return {top:1e6,bottom:1e6+200}},getContext(){return new Proxy({},{get:()=>()=>{}})}};}
function setup(extra={}){const nodes=new Map();const byId=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
 const document={getElementById:byId,querySelector:s=>byId(s.replace(/^#/,'')),querySelectorAll:()=>[],addEventListener(){},createElement:element,readyState:'complete'};
 const c=vm.createContext({console,document,window:{addEventListener(){}},setTimeout(){},requestAnimationFrame(){},location:{hash:''},innerHeight:800,Date,URL,Number,JSON,...extra});return {c,byId,nodes};}
const response=data=>({ok:true,json:async()=>data});
const flush=async()=>{for(let i=0;i<15;i++)await Promise.resolve()};
test('OpenAPI copy shows success only after clipboard confirms the exact address',async()=>{
 let resolveWrite;const writes=[];const {c,byId}=setup({navigator:{clipboard:{writeText:value=>{writes.push(value);return new Promise(resolve=>resolveWrite=resolve)}}},window:{location:{origin:'http://unit.test:8099'}}});
 vm.runInContext(section('    const btnDocs = document.getElementById("btn-app-center-docs");','    if (document.readyState === "loading")'),c);
 const pending=byId('btn-app-center-docs').fire('click');assert.equal(byId('btn-app-center-docs').disabled,true);assert(!byId('app-center-copy-status').textContent.includes('已写入'));
 assert.deepEqual(writes,['http://unit.test:8099/openapi.json']);resolveWrite();await pending;
 assert.equal(byId('btn-app-center-docs').textContent,'已复制 OpenAPI 地址');assert.match(byId('app-center-copy-status').textContent,/已写入剪贴板/);assert.equal(byId('btn-app-center-docs').disabled,false);
});
test('denied or missing clipboard never reports copied and provides the manual address',async()=>{
 for(const navigator of [{clipboard:{writeText:async()=>{throw Error('permission denied')}}},{}]){
  const {c,byId}=setup({navigator,window:{location:{origin:'http://unit.test'}}});
  vm.runInContext(section('    const btnDocs = document.getElementById("btn-app-center-docs");','    if (document.readyState === "loading")'),c);
  await byId('btn-app-center-docs').fire('click');assert.equal(byId('btn-app-center-docs').textContent,'复制失败 · 可重试');assert.match(byId('app-center-copy-status').textContent,/未写入剪贴板，请手动复制：http:\/\/unit.test\/openapi.json/);assert.equal(byId('btn-app-center-docs').disabled,false);
 }
});
function monitoring(extra={}){const env=setup(extra);env.byId('mon-asset').value='qc-01';env.byId('mon-sens').value='1.5';env.byId('mon-method').value='iqr';
 let code=scripts.find(s=>s.includes('function monitoringWindow()'));
 code=code.replace(/  runAnomaly\(\);\n  loadMonitoringEvidence\(\)\.catch[^\n]+\n/,'');
 code=code.replace(/\}\)\(\);\s*$/,'this.api={monitoringWindow,runAnomaly,runPSI,showMonitoringDetail};})();');vm.runInContext(code,env.c);return env;}
test('PSI uses explicit ISO and epoch recent windows and previous equal-length baseline',async()=>{
 const urls=[];const {c,byId}=monitoring({fetch:async url=>{urls.push(url);return response({psi:.12,level:'warn',bins:[],baseline:{n:20},recent:{n:20}})}});
 byId('mon-start').value='2026-09-01T00:00:00Z';byId('mon-end').value='2026-09-01T02:00:00Z';await c.api.runPSI();
 const u=new URL(urls[0],'http://localhost');assert.equal(u.searchParams.get('recent_start'),'2026-09-01T00:00:00.000Z');assert.equal(u.searchParams.get('baseline_start'),'2026-08-31T22:00:00.000Z');assert.equal(u.searchParams.get('baseline_end'),u.searchParams.get('recent_start'));
 byId('mon-start').value=String(Date.parse('2026-09-01T00:00:00Z')/1000);byId('mon-end').value=String(Date.parse('2026-09-01T02:00:00Z'));
 assert.equal(c.api.monitoringWindow().start,'2026-09-01T00:00:00.000Z');assert.match(byId('mon-psi-summary').textContent,/近期.*基线/);
});
test('invalid monitoring parameters make no request and failure clears stale rows and graph',async()=>{
 let calls=0;const {c,byId}=monitoring({fetch:async()=>{calls++;throw new Error('offline')}});
 byId('mon-anom-table').querySelector().innerHTML='OLD';byId('mon-sens').value='-1';await c.api.runAnomaly();assert.equal(calls,0);assert.equal(byId('mon-anom-table').querySelector().innerHTML,'');assert.match(byId('mon-anom-summary').textContent,/灵敏度/);
 byId('mon-start').value='2026-09-02';byId('mon-end').value='2026-09-01';await c.api.runPSI();assert.equal(calls,0);
 byId('mon-start').value='';byId('mon-end').value='';byId('mon-psi-table').querySelector().innerHTML='OLD';await c.api.runPSI();assert.equal(calls,1);assert.equal(byId('mon-psi-table').querySelector().innerHTML,'');assert.match(byId('mon-psi-summary').textContent,/offline/);assert.equal(byId('btn-psi').disabled,false);
});
test('monitoring parameter edits suppress old replies and detail clicks share load with last choice winning',async()=>{
 const requests=[];const {c,byId}=monitoring({fetch:url=>new Promise(resolve=>requests.push({url,resolve}))});
 const old=c.api.runPSI();await byId('mon-asset').fire('input');requests.shift().resolve(response({psi:99,bins:[]}));await old;assert.match(byId('mon-psi-summary').textContent,/参数已变化/);
 const a=c.api.showMonitoringDetail('source'),b=c.api.showMonitoringDetail('gate');assert.equal(requests.length,1);requests.shift().resolve(response({source:{mode:'fixture'},current_analysis:{admission_decision:{new_policy_suggestions_allowed:false}},boundary:{production_authority:false}}));await Promise.all([a,b]);assert.equal(JSON.parse(byId('mon-v3-detail').textContent).boundary.production_authority,false);
});
test('OpsX initial and detail requests coalesce; failures have retry feedback',async()=>{
 const requests=[];const {c,byId}=setup({fetch:()=>new Promise(resolve=>requests.push(resolve))});
 let code=scripts.find(s=>s.includes('let opsxV3Request'));
 const a=code.indexOf('  function renderOpsXV3('),b=code.indexOf('  async function loadOpsXV3',a);
 code=code.slice(0,a)+'function renderOpsXV3(j){opsxV3Evidence=j;if(selectedOpsxDetail)renderOpsXSelected();}\n'+code.slice(b);
 code=code.replace(/\}\)\(\);\s*$/,'this.loadOpsX=loadOpsXV3;})();');vm.runInContext(code,c);
 const first=byId('btn-opsx-rollout').fire('click'),second=byId('btn-opsx-security').fire('click');assert.equal(requests.length,1);requests.shift()({ok:false,status:503});await Promise.all([first,second]);assert.match(byId('opsx-v3-detail').textContent,/503/);
 const retry=byId('btn-opsx-security').fire('click');requests.shift()(response({boundary:{production_authority:false},security_contract:{fixture:true}}));await retry;assert.equal(JSON.parse(byId('opsx-v3-detail').textContent).security_contract.fixture,true);
});
test('changing Twin reliability scenario prevents stale response and releases controls',async()=>{
 let resolve;const {c,byId}=setup({getJSON:()=>new Promise(r=>resolve=r),gauge(){throw new Error('stale render')},renderErrorBars(){},renderScenarioBars(){},renderCoverageMatrix(){},fillEnvelopeTable(){}});
 c.$=s=>byId(s.slice(1));c.window.__runtimeScenarioId=x=>x;byId('tw-scen').value='strategy';
 vm.runInContext(section('  let twinRequestId=0;','  window.__loadTwinReliability='),c);
 const pending=c.loadTwin();byId('tw-scen').value='typhoon';await byId('tw-scen').fire('change');resolve({available:true});await pending;
 assert.match(byId('tw-replay-summary').textContent,/场景已切换/);assert.equal(byId('btn-tw-replay').disabled,false);
});
function twinlab(extra={}){const env=setup(extra);env.c.$=s=>env.byId(s.slice(1));env.c.$$=()=>[];env.c.ech=null;vm.runInContext(section('  let twinLabEvidence = null;','  // ========== 现场标定 =========='),env.c);return env;}
test('TwinLab report loads evidence, concurrent choice wins, and offline report never renders empty JSON',async()=>{
 const requests=[];const {c,byId}=twinlab({fetch:()=>new Promise(resolve=>requests.push(resolve))});
 const a=c.showScenarioView('contract'),b=c.showScenarioView('report');assert.equal(requests.length,1);requests.shift()(response({schema:'fixture',policy:{sha:'abc'},scenarios:{items:[]},drills:{items:[]},contracts:{items:[]}}));await Promise.all([a,b]);assert('drills' in JSON.parse(byId('scn-log').textContent));
 const fail=c.showScenarioView('run');requests.shift()({ok:false,status:503});await fail;assert.match(byId('scn-log').textContent,/失败.*503/);assert.match(byId('scn-pass').textContent,/不可用/);
});
test('TwinLab drill responses cannot replace the last selected action',async()=>{
 const requests=[];const {c,byId}=twinlab({fetch:()=>new Promise(resolve=>requests.push(resolve))});const a=c.showDrill('shadow'),b=c.showDrill('rollback');requests[1](response({enabled:false,mode:'fail_closed'}));await b;requests[0](response({selected_replay:{status:'pass',decision_count:9}}));await a;assert.match(byId('drill-state').textContent,/Rollback/);assert.match(byId('drill-state').textContent,/本页未执行回滚/);
});
test('RL button reads validated runtime decision with GET and fails closed without decision',async()=>{
 const results=[],requests=[];let data={available:true,decision:{raw_action:[.1]},policy:{model_sha256:'fixture'},telemetry:{measured:false}};
 const {c,byId}=setup({fetch:async(url,options)=>{requests.push({url,options});return response(data)},show:x=>results.push(x),selAsset:{value:'qc-01'},USE_DRIVERS:false});c.$=s=>byId(s.slice(1));
 vm.runInContext(section('    let strategyActionId=0;','        // 作业驱动开关'),c);await c.readStrategyAction('rl');
 assert.equal(requests[0].url,'/api/v3/runtime/frame');assert(!requests[0].options.body);assert.equal(results.at(-1).decision.raw_action[0],.1);assert.equal(results.at(-1).production_authority,false);
 data={available:true,policy:{}};await c.readStrategyAction('rl');assert.match(results.at(-1).detail,/不生成替代建议/);assert.equal(byId('btn-rl').disabled,false);
});
test('aggregate request uses selected scenario and removes old graph on HTTP failure',async()=>{
 const draws=[],urls=[];const {c,byId}=setup({fetch:async url=>{urls.push(url);return {ok:false,status:503}},drawAgg:x=>draws.push(x),runtimeScenarioId:()=> 'typhoon',show(){}});c.$=s=>byId(s.slice(1));byId('sel-scenario').value='台风扰动';
 vm.runInContext(section('    let aggregateRequestId=0;','    /************** 3D 场景（保持）'),c);await c.runAggregateScenario();assert.match(urls[0],/scenario=typhoon/);assert.equal(draws.at(-1).p50.length,0);assert.match(byId('aggLegend').textContent,/旧曲线已清除.*503/);assert.equal(byId('btn-agg-sim').disabled,false);
});
test('3D layer changed during a pending refresh gets its own refresh',async()=>{
 const pending=[];const {c}=setup({VIEW_MODE:'now',OVERLAY_CARBON:false,TWIN_REFRESH_IN_FLIGHT:null,refresh3DColorsOnce:()=>new Promise(resolve=>pending.push(resolve))});vm.runInContext(section('    async function refresh3DColors(){','    async function refresh3DColorsOnce(){'),c);
 const a=c.refresh3DColors();c.OVERLAY_CARBON=true;const b=c.refresh3DColors();assert.equal(pending.length,1);pending.shift()(['old']);await a;await flush();assert.equal(pending.length,1);pending.shift()(['carbon']);assert.deepEqual(await b,['carbon']);
});
test('asset selection prevents previous forecast from replacing current curve or reopening old stream',async()=>{
 const requests=[],dashlets=[];let closed=0;const {c,byId}=setup({assetRequestId:0,VIEW_MODE:'forecast',USE_DRIVERS:false,sse:{close(){closed++}},lineData:[1],fcstData:[1],selAsset:{value:'A'},drawLine(){},resetRealtimeInsights(){},renderSelChart(){},toForecastEnvelope:x=>x,loadDashlets:x=>dashlets.push(x),fetch:()=>new Promise(resolve=>requests.push(resolve))});c.$=s=>byId(s.slice(1));
 vm.runInContext(section('    let assetRequestId=0;','    /************** 顶部视图/芯片'),c);c.switchAsset('A');c.selAsset.value='B';c.switchAsset('B');requests[1](response({points:[{ts:'new',p50:2}]}));await flush();requests[0](response({points:[{ts:'old',p50:1}]}));await flush();assert.equal(c.fcstData[0].p50,2);assert.deepEqual(dashlets,['B']);assert.equal(closed,1);
});
test('switching context clears old risk/value cards and invalidates their pending request',()=>{
 const {c,byId}=setup({dashletsRequestId:8,renderDashlets(){},renderDashletsX(){},calibStats:{mape:12},peakRisk:{m15:.9},savings:{cny:100},dqStats:{missing:.5}});
 vm.runInContext(section('    function resetRealtimeInsights(message){','    function loadDashlets('),c);c.resetRealtimeInsights('loading new asset');
 assert.equal(c.dashletsRequestId,9);assert.equal(c.calibStats.mape,null);assert.equal(c.peakRisk.m15,null);assert.equal(c.savings.cny,null);assert.equal(c.dqStats.missing,null);assert.equal(byId('realtimeBasis').textContent,'loading new asset');
});
