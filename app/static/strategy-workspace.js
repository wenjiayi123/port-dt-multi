/* Reorganize the existing strategy controls; no model, simulator or gateway copy. */
(() => {
  'use strict';
  const root=document.getElementById('strategy-exec-module');
  if(!root)return;
  const $=id=>document.getElementById(id);
  const active=()=>document.body.dataset.moduleView===root.id;
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const numeric=n=>typeof n==='number'&&Number.isFinite(n);
  const fmt=(n,d=2)=>numeric(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
  let shell,tab='scene',scheduled=false,sim=null,aggregate=null,capability=null,capabilityError='',assetRequest;
  window.PortStrategyCharts=Object.freeze({draw(canvas,series,options={}){
    if(!canvas.getClientRects().length)return false;
    const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),width=rect.width,height=rect.height||280;
    if(width<80)return false;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,width,height);c.fillStyle='#10222e';c.fillRect(0,0,width,height);canvas.setAttribute('aria-label',options.label||'功率对比 kW');
    const valid=series.length&&series[0].values?.length>=2&&series.every(s=>s.values.length===series[0].values.length&&s.values.every(numeric));if(!valid)return false;
    const band=options.low?.length===series[0].values.length&&options.high?.length===series[0].values.length&&options.low.every(numeric)&&options.high.every(numeric);
    const all=series.flatMap(s=>s.values).concat(band?[...options.low,...options.high]:[]),lo=Math.min(...all),hi=Math.max(...all),pad=(hi-lo)*.13||Math.max(1,hi*.08),min=lo-pad,max=hi+pad;
    const left=65,right=23,top=40,bottom=37,w=width-left-right,h=height-top-bottom,n=series[0].values.length,x=i=>left+i/(n-1)*w,y=v=>top+h-(v-min)/(max-min||1)*h;
    c.font='10px system-ui';c.textAlign='right';c.lineWidth=1;
    for(let i=0;i<5;i++){const value=min+(max-min)*i/4,py=y(value);c.strokeStyle='#28424f';c.beginPath();c.moveTo(left,py);c.lineTo(width-right,py);c.stroke();c.fillStyle='#7e9ead';c.fillText(fmt(value,0),left-10,py+3);}
    c.textAlign='left';c.fillStyle='#7e9ead';c.fillText('kW',left,17);
    c.textAlign='center';for(let i=0;i<=5;i++){const at=Math.round((n-1)*i/5),time=options.times?.[at],label=time&&Number.isFinite(Date.parse(time))?new Date(time).toLocaleTimeString('zh-CN',{timeZone:'Asia/Shanghai',hour:'2-digit',minute:'2-digit',hour12:false}):String(at+1);c.fillText(label,x(at),height-19);}c.textAlign='right';c.fillStyle='#5e8394';c.fillText(options.xLabel||'时间步',width-right,height-3);
    if(band){c.beginPath();options.high.forEach((v,i)=>i?c.lineTo(x(i),y(v)):c.moveTo(x(i),y(v)));for(let i=n-1;i>=0;i--)c.lineTo(x(i),y(options.low[i]));c.closePath();c.fillStyle='#66c7b21b';c.fill();}
    series.forEach((s,i)=>{c.strokeStyle=s.color;c.lineWidth=1.8;c.beginPath();s.values.forEach((v,j)=>j?c.lineTo(x(j),y(v)):c.moveTo(x(j),y(v)));c.stroke();const px=left+90+i*110;c.fillStyle=s.color;c.fillRect(px,10,13,2);c.textAlign='left';c.fillStyle='#94b4c0';c.fillText(s.name,px+19,15);});return true;
  }});
  function move(id,target){const node=$(id),host=$(target);if(node&&host)host.append(node);return node;}
  function group(node,target){if(node&&$(target))$(target).append(node);}
  function details(title,id){const el=document.createElement('details');el.className='sw-details';el.innerHTML=`<summary>${esc(title)}</summary><div id="${id}" class="sw-detail-content"></div>`;return el;}
  function mount(){
    if(shell)return;
    shell=document.createElement('section');shell.className='sw-workspace';shell.dataset.i18nSkip='';shell.setAttribute('aria-label','策略推演与评测工作台');
    shell.innerHTML=`<div class="sw-heading"><div><div class="sw-eyebrow">STRATEGY WORKSPACE / DECISION SUPPORT</div><h2>策略推演与评测</h2><p>观察场景趋势，比较登记策略，再核验执行条件。</p></div><div class="sw-heading-tools"><span class="sw-boundary"><i></i>离线评测 · 设备控制未授权</span><details class="sw-more"><summary>更多工具 ↗</summary><div id="sw-tool-links"></div></details></div></div>
      <div class="sw-nav" role="tablist" aria-label="策略工作区"><button type="button" role="tab" id="sw-tab-scene" data-sw-tab="scene" aria-controls="sw-panel-scene" aria-selected="true"><b>01</b><span>场景推演<small>查看运行趋势</small></span></button><button type="button" role="tab" id="sw-tab-evaluate" data-sw-tab="evaluate" aria-controls="sw-panel-evaluate" aria-selected="false"><b>02</b><span>策略评测<small>比较留出集结果</small></span></button><button type="button" role="tab" id="sw-tab-verify" data-sw-tab="verify" aria-controls="sw-panel-verify" aria-selected="false"><b>03</b><span>执行核验<small>查看接入与授权</small></span></button><span class="sw-nav-status" id="sw-live-status" role="status">选择场景后运行推演</span></div>
      <section id="sw-panel-scene" role="tabpanel" aria-labelledby="sw-tab-scene" class="sw-panel"><div class="sw-scene-grid"><aside class="sw-card sw-config"><div class="sw-section-number">SCENARIO / 01</div><h3>场景配置</h3><p>使用当前校准回放与已校验策略运行时。</p><label class="sw-field" for="sel-scenario">推演场景</label><div id="sw-scene-select"></div><div class="sw-window"><span>展示窗口<b>未来 6 小时</b></span><span>展示采样<b>1 分钟</b></span></div><div id="sw-run-scenario"></div><div class="sw-note">曲线用于观察所选场景；下一页的模型评测使用各自登记的独立留出集。</div><button type="button" class="sw-secondary" data-sw-go="evaluate">进入策略评测 →</button></aside><div class="sw-card sw-scenario-chart"><div class="sw-card-head"><div><h3>场景负荷趋势</h3><p id="sw-scenario-name">基线场景</p></div><div class="sw-chart-key"><i></i>P50<span>P10–P90 区间</span></div></div><div class="sw-chart-stage" id="sw-aggregate-stage"><div class="sw-empty" id="sw-aggregate-empty"><span>⌁</span><b>等待场景推演</b><small>点击“运行场景推演”生成有来源的负荷曲线</small></div></div><div id="sw-aggregate-legend" class="sw-chart-caption"></div><div id="sw-aggregate-facts" class="sw-aggregate-facts"></div></div></div><div id="sw-extra-slot"></div></section>
      <section id="sw-panel-evaluate" role="tabpanel" aria-labelledby="sw-tab-evaluate" class="sw-panel" hidden><div class="sw-evaluate-grid"><aside class="sw-card sw-candidates"><div class="sw-card-head"><div><div class="sw-section-number">REGISTRY / 02</div><h3>登记策略</h3></div><div id="sw-list-action"></div></div><div class="sw-field"><label for="rlpanel-strategy">当前策略</label><div id="sw-native-select"></div></div><div class="sw-policy-list" id="sw-policy-list"><div class="sw-empty"><span>◇</span><b>尚未读取策略</b><small>从模型登记库读取可核验的候选，不预填收益。</small></div></div><div id="sw-selection-hint"></div><div id="sw-policy-evidence"></div></aside><div class="sw-result-column"><div class="sw-card sw-results"><div class="sw-card-head"><div><h3>留出集评测对比</h3><p id="sw-evaluation-state">选择一条登记策略后运行评测</p></div><div id="sw-evaluate-action"></div></div><div class="sw-metrics" id="sw-eval-metrics"></div><div class="sw-chart-stage sw-comparison" id="sw-comparison-stage"><div class="sw-empty" id="sw-evaluation-empty"><span>⌁</span><b>尚无当前策略的评测结果</b><small>评测完成后展示基线与策略功率，切换策略会清除旧结果。</small></div></div><div id="sw-sim-meta" class="sw-chart-caption"></div><div class="sw-result-actions"><div id="sw-save-action"></div><button type="button" class="sw-secondary" data-sw-go="verify">查看执行条件 →</button></div></div><div id="sw-model-evidence"></div></div></div></section>
      <section id="sw-panel-verify" role="tabpanel" aria-labelledby="sw-tab-verify" class="sw-panel" hidden><div class="sw-verification-grid"><div class="sw-card sw-permission"><div class="sw-section-number">ADMISSION / 03</div><h3>先核验条件，再进入执行工作区</h3><div id="sw-permission-summary"></div><div class="sw-gates"><div><i>1</i><b>设备白名单</b><small>来源与设备身份</small></div><div><i>2</i><b>工程约束</b><small>限值、联锁与时效</small></div><div><i>3</i><b>异人确认</b><small>独立人员复核</small></div><div><i>4</i><b>审计与回滚</b><small>回执和恢复路径</small></div></div><p class="sw-note">以上是执行流程要求，状态以网关实际返回为准。评测通过也不会在本页授予或下发设备指令。</p><div class="sw-verify-actions" id="sw-verify-links"></div><div id="sw-dispatch-evidence"></div></div><aside class="sw-card sw-gateway"><div class="sw-card-head"><div><h3>现场接入能力</h3><p>读取既有南向网关</p></div><div id="sw-gateway-refresh"></div></div><div class="sw-table-scroll" id="sw-gateway-table"></div><div id="sw-rollout-slot"></div></aside></div></section><div id="sw-receipt-slot"></div>`;
    root.prepend(shell);
    move('sel-scenario','sw-scene-select');move('btn-agg-sim','sw-run-scenario').textContent='运行场景推演';
    move('aggLine','sw-aggregate-stage');move('aggLegend','sw-aggregate-legend');
    move('btn-rlpanel-list','sw-list-action').textContent='读取策略';move('rlpanel-strategy','sw-native-select');
    move('btn-rlpanel-sim','sw-evaluate-action').textContent='运行留出集评测';
    move('strategy-selection-hint','sw-selection-hint');
    const windowNote=document.createElement('p');windowNote.id='sw-evaluation-window';windowNote.className='sw-result-window';$('sw-eval-metrics').before(windowNote);
    move('simCanvas','sw-comparison-stage');move('simMeta','sw-sim-meta');move('btn-sim-save','sw-save-action').textContent='导出当前对比图';
    for(const id of ['btn-rlpanel-open','btn-rl-future-open','btn-strategy-to-twin','btn-strategy-open-rl'])move(id,'sw-tool-links');
    // Keep one RL entry in the menu; both original handlers and IDs remain intact.
    $('btn-strategy-open-rl').hidden=true;
    const policyEvidence=details('策略依据与来源','sw-policy-evidence-body');$('sw-policy-evidence').append(policyEvidence);
    group($('strategy-why-head').closest('.twin-core-card'),'sw-policy-evidence-body');group($('strategy-why-list').closest('.strategy-chain-detail'),'sw-policy-evidence-body');
    const modelEvidence=details('模型证据与归因','sw-model-evidence-body');$('sw-model-evidence').append(modelEvidence);
    move('btn-rlpanel-explain','sw-model-evidence-body').textContent='读取模型证据与解释';
    group($('expBar').closest('.exp-wrap'),'sw-model-evidence-body');
    group($('strategy-sim-head').closest('.twin-core-card'),'sw-model-evidence-body');group($('strategy-sim-list').closest('.strategy-chain-detail'),'sw-model-evidence-body');
    group($('strategy-ready-head').closest('.twin-core-card'),'sw-permission-summary');
    move('btn-strategy-to-exec','sw-verify-links').textContent='打开执行核验工作区 ↗';move('btn-strategy-to-audit','sw-verify-links').textContent='查看治理与审计 ↗';
    const boundary=details('执行边界依据','sw-dispatch-evidence-body');$('sw-dispatch-evidence').append(boundary);group($('strategy-ready-list').closest('.strategy-chain-detail'),'sw-dispatch-evidence-body');
    move('btn-rlpanel-history','sw-gateway-refresh').textContent='刷新能力';
    const gatewayDetails=details('网关原始能力与模型上线信息','sw-gateway-raw');$('sw-gateway-table').append(gatewayDetails);move('tbl-history','sw-gateway-raw');move('sw-rollout-slot','sw-gateway-raw');
    const gatewayOverview=document.createElement('div');gatewayOverview.id='sw-gateway-overview';$('sw-gateway-table').before(gatewayOverview);
    const extra=details('补充查询：资产预测、运行状态与简报','sw-extra-body');$('sw-extra-slot').append(extra);
    $('sw-extra-body').innerHTML='<div class="sw-extra-row"><label for="sw-asset-select">查询资产</label><select id="sw-asset-select"><option value="">展开后读取资产</option></select><span id="sw-assets-status" role="status"></span></div><div class="sw-extra-row" id="sw-driver-row"></div><div class="sw-extra-row" id="sw-extra-actions"></div><p class="sw-note">作业驱动开关仅用于资产预测。查询结果保存在页面底部的接口回执中。</p>';
    group($('chkDrivers').closest('label'),'sw-driver-row');
    for(const [id,label] of [['btn-forecast','查询资产预测'],['btn-rl','查询当前策略状态'],['btn-twin','查询当前孪生状态'],['btn-report','查询资产简报']])move(id,'sw-extra-actions').textContent=label;
    $('btn-forecast').disabled=$('btn-report').disabled=true;
    const receipt=details('接口回执与技术来源','sw-receipt-body');receipt.id='sw-receipt';$('sw-receipt-slot').append(receipt);$('sw-receipt-body').innerHTML='<p class="sw-note">保留最近一次接口回执。切换策略后，是否已有当前结果以上方评测状态为准。</p>';move('out','sw-receipt-body');
    const context=details('上游上下文','sw-upstream-body');$('sw-receipt-body').append(context);move('strategy-origin-sub','sw-upstream-body');group(root.querySelector('.chain-context-tags'),'sw-upstream-body');
    extra.addEventListener('toggle',()=>{if(extra.open)loadAssets();});
    modelEvidence.addEventListener('toggle',()=>{if(modelEvidence.open)requestResize();});
    shell.addEventListener('click',event=>{
      const control=event.target.closest('[data-sw-tab],[data-sw-go],[data-sw-policy]');
      if(control?.dataset.swTab||control?.dataset.swGo)selectTab(control.dataset.swTab||control.dataset.swGo);
      if(control?.dataset.swPolicy){const select=$('rlpanel-strategy');select.value=control.dataset.swPolicy;select.dispatchEvent(new Event('change',{bubbles:true}));schedule();}
      if(event.target.closest('#btn-rlpanel-explain'))modelEvidence.open=true;
      if(event.target.closest('#sw-extra-actions'))receipt.open=true;
    });
    shell.querySelector('.sw-nav').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;const tabs=['scene','evaluate','verify'];let i=tabs.indexOf(tab);i=event.key==='Home'?0:event.key==='End'?2:(i+(event.key==='ArrowRight'?1:2))%3;event.preventDefault();selectTab(tabs[i]);$('sw-tab-'+tabs[i]).focus();});
    const observer=new MutationObserver(schedule);
    for(const node of [$('rlpanel-strategy'),$('btn-rlpanel-sim'),$('simMeta'),$('aggLegend'),$('out')])observer.observe(node,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['disabled','data-payload']});
    const legacy=root.querySelector('.strategy-rl-card');if(legacy)new MutationObserver(schedule).observe(legacy,{childList:true});
    $('sel-scenario').addEventListener('change',()=>{aggregate=null;schedule();});
    $('rlpanel-strategy').addEventListener('change',()=>{sim=null;schedule();});
    render();selectTab(tab);
  }
  function requestResize(){requestAnimationFrame(()=>{if(active())window.dispatchEvent(new Event('resize'));});}
  function selectTab(next){tab=next;for(const key of ['scene','evaluate','verify']){$('sw-panel-'+key).hidden=key!==next;const button=$('sw-tab-'+key);button.setAttribute('aria-selected',String(key===next));button.tabIndex=key===next?0:-1;}shell.dataset.tab=next;render();requestResize();}
  function schedule(){if(!active()||scheduled||!shell)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;if(active())render();});}
  function policies(){try{return JSON.parse($('rlpanel-strategy').dataset.payload||'[]');}catch(_){return [];}}
  function render(){
    if(!shell)return;
    const list=policies(),selected=$('rlpanel-strategy').value;
    const key=JSON.stringify([selected,list.map(p=>[p.id,p.title,p.meta?.algorithm,p.meta?.dataset_id,p.meta?.artifact_verified])]);
    if($('sw-policy-list').dataset.renderKey!==key){$('sw-policy-list').dataset.renderKey=key;$('sw-policy-list').innerHTML=list.length?list.map(p=>`<button type="button" data-sw-policy="${esc(p.id)}" aria-pressed="${String(p.id)===selected}" class="sw-policy ${String(p.id)===selected?'selected':''}"><span class="sw-policy-name">${esc((p.meta?.algorithm||p.title||p.id).toUpperCase())}<i>${String(p.id)===selected?'已选择':'选择'}</i></span><span class="sw-policy-dataset">${esc(p.meta?.dataset_id||'数据集未登记')}</span><small>${p.meta?.artifact_verified===true?'制品已校验':'制品待核验'} · ${esc(p.id)}</small></button>`).join(''):'<div class="sw-empty"><span>◇</span><b>尚无可选策略</b><small>点击“读取策略”；失败可重试，缺失结果不预填。</small></div>';}
    if(sim&&String(sim.strategy_id)!==selected)sim=null;
    const metrics=sim?.evaluation?.metrics||{},s=sim?.summary||{},episodes=sim?.evaluation?.episodes;
    const values=[['电量差',s.delta_kWh,'kWh · 本次对比窗：基线 − 策略'],['峰值差',s.peak_reduction_kW,'kW · 本次对比窗：基线 − 策略'],['单位吞吐成本',metrics.cost_per_teu,`¥/TEU · ${episodes??'—'}回合评测均值`],['服务完成率',numeric(metrics.service_completion_ratio)?metrics.service_completion_ratio*100:null,`% · ${episodes??'—'}回合评测均值`]];
    const metricHTML=values.map(([label,value,unit])=>`<div class="sw-metric"><span>${label}</span><strong class="${numeric(value)&&value<0?'negative':''}">${fmt(value,2)}</strong><small>${unit}</small></div>`).join('');if($('sw-eval-metrics').innerHTML!==metricHTML)$('sw-eval-metrics').innerHTML=metricHTML;
    $('sw-evaluation-empty').hidden=!!sim;$('simCanvas').hidden=!sim;
    const e=sim?.evaluation;$('sw-evaluation-state').textContent=e?`${String(e.algorithm||'').toUpperCase()} · ${e.dataset_id||'—'} · ${e.split||'留出集'} · ${e.episodes??'—'} 回合`:'选择登记策略后评测；评测数据独立于场景推演';
    const period=s.window,stamp=value=>Number.isFinite(Date.parse(value))?new Date(value).toISOString().replace('T',' ').slice(0,16):'未提供';$('sw-evaluation-window').textContent=period?`曲线对比窗：${stamp(period.start)} → ${stamp(period.end)} UTC；同窗基线与策略对比。`:'';
    const selectedPolicy=list.find(p=>String(p.id)===selected);$('strategy-selection-hint').textContent=selectedPolicy?`登记数据集：${selectedPolicy.meta?.dataset_id||'未登记'}。当前场景不改变该模型的独立留出集。`:'候选来自模型登记库；读取后查看算法、数据集与制品状态。';
    const pending=!sim&&selected&&$('btn-rlpanel-sim').disabled&&!$('btn-rlpanel-list').disabled,failed=!sim&&/失败/.test($('simMeta').textContent);
    $('sw-evaluation-empty').querySelector('b').textContent=pending?'正在运行留出集评测':failed?'本次评测未完成':'尚无当前策略的评测结果';
    $('sw-evaluation-empty').querySelector('small').textContent=pending?'正在读取模型并计算真实结果；可以切换工作区查看接入条件。':failed?$('simMeta').textContent.slice(0,200):'评测完成后展示基线与策略功率；切换策略会清除旧结果。';
    $('sw-scenario-name').textContent=$('sel-scenario').value+' · 6小时运行时推演';
    $('sw-aggregate-empty').hidden=!!aggregate;$('aggLine').hidden=!aggregate;
    const summary=aggregate?.summary;
    $('sw-aggregate-facts').textContent=summary?`峰值 ${fmt(summary.peak_kw,1)} kW　·　窗口电量 ${fmt(summary.energy_kwh,1)} kWh　·　${summary.decision_count??'—'} 次策略决策　·　软件硬约束 ${summary.hard_guardrail_passed===true?'通过':summary.hard_guardrail_passed===false?'未通过':'待核验'}`:'';
    const status=tab==='scene'?aggregate?'场景推演已返回 · 非现场实况':$('aggLegend').textContent:tab==='evaluate'?sim?'评测结果就绪 · 不授予执行权':pending?'正在评测当前策略…':failed?'评测未完成 · 可重试':list.length?`已读取 ${list.length} 条登记策略`:'先读取候选策略，再运行评测':'执行条件以网关返回为准';$('sw-live-status').textContent=status;
    if(!aggregate){$('sw-aggregate-empty').querySelector('b').textContent=$('btn-agg-sim').disabled?'正在推演所选场景':/失败/.test($('aggLegend').textContent)?'场景推演未完成':'等待场景推演';}
    const rollout=$('rolloutCard');if(rollout&&rollout.parentElement!==$('sw-rollout-slot'))$('sw-rollout-slot').append(rollout);
    const cap=capability,yesNo=(value,yes,no)=>value===true?yes:value===false?no:'待核验';
    const capHTML=[['网关启用',cap?yesNo(cap.enabled,'已启用','未启用'):capabilityError?'读取失败 · 请刷新':'等待读取'],['白名单设备',cap?fmt(cap.whitelisted_asset_count,0)+' 台':'—'],['确认机制',cap?yesNo(cap.two_person_confirmation_required,'要求异人确认','待核验'):'—'],['现场执行验收',cap?yesNo(cap.site_execution_accepted,'已验收','待验收'):'—']].map(([k,v])=>`<div class="sw-cap-stat"><small>${k}</small><b>${v}</b></div>`).join('');
    if($('sw-gateway-overview').innerHTML!==capHTML)$('sw-gateway-overview').innerHTML=capHTML;
  }
  async function loadAssets(){
    if(assetRequest)return assetRequest;
    $('sw-assets-status').textContent='正在读取…';
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),12000);
    assetRequest=fetch('/api/assets',{cache:'no-store',signal:controller.signal}).then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();}).then(list=>{
      if(!Array.isArray(list)||!list.length)throw new Error('资产列表为空');const select=$('sw-asset-select');select.replaceChildren(...list.map(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.label||item.id;return option;}));$('sw-assets-status').textContent='当前资产用于本区查询';$('btn-forecast').disabled=$('btn-report').disabled=false;render();
    }).catch(error=>{assetRequest=null;$('sw-assets-status').textContent=error.message+' · 收起后展开可重试';}).finally(()=>clearTimeout(timer));
    return assetRequest;
  }
  window.addEventListener('port-strategy-result',event=>{sim=event.detail?.type==='evaluated'?event.detail.result:null;schedule();requestResize();});
  window.addEventListener('port-strategy-aggregate',event=>{aggregate=event.detail?.result||null;schedule();requestResize();});
  window.addEventListener('port-strategy-capability',event=>{capability=event.detail?.capability||null;capabilityError=event.detail?.error||'';schedule();});
  function route(){if(active()){mount();document.body.classList.add('mv-desktop','sw-desktop');render();requestResize();}else document.body.classList.remove('sw-desktop');}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',route,{once:true});else route();
  window.addEventListener('port-module-shown',route);
})();
