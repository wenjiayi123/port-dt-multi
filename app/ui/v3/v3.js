const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat('zh-CN');
const metricNames = {throughput_teu:'吞吐',delay_index_mean:'延误',energy_cost:'能耗成本',carbon_kg:'碳排',peak_kw:'峰值',cost_per_teu:'单位吞吐成本',carbon_kg_per_teu:'单位吞吐碳强度',energy_kwh_per_teu:'单位吞吐能耗',grid_energy_kwh:'电网电量',service_completion_ratio:'作业完成率',queue_peak_teu:'队列峰值',queue_end_teu:'期末队列',bess_equivalent_full_cycles:'储能等效循环',weather_block_rate:'气象封锁率',action_projection_rate:'动作安全投影率',action_projection_correction_kw_mean:'平均动作修正',action_projection_severity_mean:'平均修正严重度',action_projection_grid_cap_rate:'电网容量修正率',action_projection_soc_bound_rate:'SOC边界修正率',action_projection_terminal_reachability_rate:'期末SOC可达域修正率',action_projection_power_bound_rate:'功率/爬坡修正率',operational_resource_factor_mean:'资源可用因子',service_factor_mean:'服务强度',guardrail_violation_rate:'安全违规率',terminal_soc_error:'期末 SOC 误差',bess_throughput_kwh:'储能吞吐电量',flex_shift_energy_kwh:'柔性负荷移峰电量'};
const stateNames = {implemented:'已实现',contract_ready:'契约就绪',adapter_required:'需适配器'};
const metricUnits = {throughput_teu:'TEU / 48h',delay_index_mean:'指数',energy_cost:'CNY / 48h',carbon_kg:'kgCO₂ / 48h',peak_kw:'kW',cost_per_teu:'CNY/TEU',carbon_kg_per_teu:'kgCO₂/TEU',energy_kwh_per_teu:'kWh/TEU',grid_energy_kwh:'kWh',service_completion_ratio:'比例',queue_peak_teu:'TEU',queue_end_teu:'TEU',bess_equivalent_full_cycles:'EFC',weather_block_rate:'比例',action_projection_rate:'比例',action_projection_correction_kw_mean:'kW/步',action_projection_severity_mean:'额定功率比例',action_projection_grid_cap_rate:'比例',action_projection_soc_bound_rate:'比例',action_projection_terminal_reachability_rate:'比例',action_projection_power_bound_rate:'比例',operational_resource_factor_mean:'比例',service_factor_mean:'比例',guardrail_violation_rate:'比例',terminal_soc_error:'SOC',bess_throughput_kwh:'kWh',flex_shift_energy_kwh:'kWh'};
let overviewData = null;
let readinessData = null;
let impactRenderGeneration = 0;

function safeNumber(value, digits=1){
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}
function escapeHTML(value){ return String(value??'').replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }
function pct(value,digits=2){ const n=Number(value); return Number.isFinite(n)?`${n>=0?'+':''}${(100*n).toFixed(digits)}%`:'—'; }
function metricValue(name,value){
  const n=Number(value); if(!Number.isFinite(n)) return '—';
  if(['guardrail_violation_rate','service_completion_ratio','weather_block_rate','action_projection_rate','action_projection_severity_mean','action_projection_grid_cap_rate','action_projection_soc_bound_rate','action_projection_terminal_reachability_rate','action_projection_power_bound_rate','operational_resource_factor_mean'].includes(name)) return `${(100*n).toFixed(2)}%`;
  if(name==='delay_index_mean') return n.toFixed(2);
  return new Intl.NumberFormat('zh-CN',{maximumFractionDigits:name==='energy_cost'||name==='carbon_kg'?0:1}).format(n);
}
function toast(message){ const el=$('toast'); el.textContent=message; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),4200); }
function setClock(){ $('clock').textContent=new Date().toLocaleTimeString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'})+' CST'; }
setClock(); setInterval(setClock,1000);

function renderAlgorithms(rows){
  const familyColor={RL:'#6fe4cf',Control:'#f2b45c',Rule:'#62a8e7'};
  $('algorithmGrid').innerHTML=rows.map((row,index)=>`<article class="algo-card" style="--family-color:${familyColor[row.family]||'#6fe4cf'}">
    <div class="algo-top"><span class="algo-id">A-${String(index+1).padStart(2,'0')} / ${row.family.toUpperCase()}</span><b class="algo-state">${row.formal_runs ? row.formal_runs+' 正式运行':'实现已接通'}</b></div>
    <h3>${row.name}</h3><p>${row.description}</p>
    <div class="algo-meta"><span>${row.action_space==='continuous'?'连续动作':'离散动作'}</span><b>${row.trainable?'真实优化器':'确定性基线'}</b></div>
    <button class="card-action" type="button" data-algorithm="${escapeHTML(row.id)}">查看训练指标 <span>↗</span></button>
  </article>`).join('');
}

function renderCapabilities(rows){
  $('capabilityGrid').innerHTML=rows.map((row,index)=>`<article class="capability-card">
    <span class="cap-no">OPS-${String(index+1).padStart(2,'0')}</span><i class="cap-state ${row.state}" title="${stateNames[row.state]}"></i>
    <h3>${row.name}</h3><p>${row.engine}</p><small>${escapeHTML(row.depth?.implementation_label||stateNames[row.state])} · ${row.depth?.model_output_available?'有运行输出':'无独立模型输出'}</small><small>现场替换 · ${row.site_replacement}</small>
    <button class="card-action" type="button" data-capability="${escapeHTML(row.id)}">查看技术链路 <span>↗</span></button>
  </article>`).join('');
  const coverage=overviewData?.business_domain_coverage||{};
  $('domainCoverageSummary').innerHTML=`<span>${fmt.format(coverage.domain_count||0)} 个业务域</span><span>${fmt.format(coverage.runtime_output_available_count||0)} 个有运行输出</span><span>${fmt.format(coverage.no_independent_optimizer_count||0)} 个无独立优化器</span><span>代码哈希 ${coverage.all_code_artifacts_hash_verified?'全部通过':'存在缺失'}</span><span>生产准入 ${fmt.format(coverage.production_ready_count||0)}</span>`;
}

function renderGates(rows){
  $('gateFlow').innerHTML=rows.map((row,index)=>`<article class="gate-item ${row.state==='software_ready'?'ready':''}">
    <span class="gate-num">GATE ${String(index+1).padStart(2,'0')}</span><div class="gate-icon">${row.state==='software_ready'?'✓':'◇'}</div>
    <h3>${row.name}</h3><span>${row.state==='software_ready'?'软件能力已具备':'现场部署前必须完成'}</span><button class="gate-action" type="button" data-gate="${escapeHTML(row.id)}">验收规则 ↗</button>
  </article>`).join('');
}

function renderAdvantage(payload){
  if(!payload || !payload.selected) return;
  const selected=payload.selected;
  const mean=Number(selected.weighted_relative_improvement?.mean);
  const pct=Number.isFinite(mean)?mean*100:null;
  $('advantageValue').textContent=pct===null?'—':`${pct>=0?'+':''}${pct.toFixed(2)}%`;
  $('advantageLabel').textContent=payload.claim_status==='STRICT_ADVANTAGE_95CI'?'95% CI 严格优势成立':payload.claim_status==='TEST_SAFETY_ADMISSION_FAILED'?'盲测安全门未通过，不成立优势声明':'点估计领先，95% CI 尚未确认';
  $('winnerName').textContent=selected.name;
  $('winnerImpl').textContent=selected.implementation;
  $('winnerScore').textContent=pct===null?'—':`${pct>=0?'+':''}${pct.toFixed(1)}%`;
  const arc=Math.min(302,Math.max(0,Math.abs(pct||0)*5)); $('scoreArc').style.strokeDasharray=`${arc} 302`;
  const claim=$('claimStatus'); claim.textContent=payload.claim_status.replaceAll('_',' '); claim.className=selected.strict_advantage?'verified':'pending';
  const entries=Object.entries(selected.metrics_relative_to_fcfs||{});
  const max=Math.max(0.01,...entries.map(([,v])=>Math.abs(Number(v.mean)||0)));
  $('advantageBars').innerHTML=entries.map(([name,summary])=>{const value=Number(summary.mean)||0;const height=Math.max(3,Math.abs(value)/max*65);return `<div class="adv-bar ${value<0?'neg':''}"><b>${value>=0?'+':''}${(value*100).toFixed(1)}%</b><div class="bar-track"><i style="height:${height}px"></i></div><span>${metricNames[name]||name}</span></div>`}).join('');
}

function renderImpact(view='rl'){
  document.querySelectorAll('[data-impact-view]').forEach(button=>button.classList.toggle('active',button.dataset.impactView===view));
  if(!overviewData) return;
  const generation=++impactRenderGeneration;
  const metrics=$('impactMetrics'); const advantage=overviewData.advantage; const impact=overviewData.business_impact;
  if(view==='twin'){
    $('impactStatus').textContent='SOFTWARE STRESS / SITE FIDELITY SEPARATED'; $('impactHeadline').textContent='正在执行孪生可靠性检查';
    $('impactDescription').textContent='软件压力回放与现场保真度分开评定；未接入实测结果的项目不自动补值。';
    metrics.innerHTML='<div class="skeleton wide"></div>';
    $('impactBoundary').textContent='正在从后端执行可复现压力场景。';
    fetch('/api/v3/twin/reliability?refresh=1&scenario=typhoon_closure',{cache:'no-store'}).then(response=>{if(!response.ok) throw new Error(`twin reliability ${response.status}`);return response.json();}).then(payload=>{
      if(generation!==impactRenderGeneration) return;
      const coverage=payload.software_coverage||{}; const stress=payload.software_stress||{}; const replay=payload.selected_replay||{};
      $('impactStatus').textContent='BOUNDED STRESS / RECOMMENDATION ONLY';
      $('impactHeadline').textContent=`软件压力场景 ${stress.passed||0}/${stress.total||0} 通过 · 现场校准待接入`;
      $('impactDescription').textContent=`当前选中“${replay.name||replay.id||'台风封闭'}”回放；策略输出仅建议，无执行副作用。压力场景通过不代表现场事故频率或孪生保真度已验证。`;
      metrics.innerHTML=`<article class="impact-kpi positive"><span>孪生软件场景覆盖</span><strong>${fmt.format(coverage.covered||0)} / ${fmt.format(coverage.total||0)}</strong><small>代码路径可执行，非现场保真度</small></article><article class="impact-kpi ${stress.passed===stress.total?'positive':'negative'}"><span>有界压力测试</span><strong>${fmt.format(stress.passed||0)} / ${fmt.format(stress.total||0)}</strong><small>哈希绑定策略 · 失效安全统计</small></article><article class="impact-kpi ${replay.passed?'positive':'negative'}"><span>台风封闭回放</span><strong>${replay.passed?'通过':'未通过'}</strong><small>违规 ${fmt.format(replay.violation_count||0)} · side effect ${escapeHTML(replay.side_effect||'—')}</small></article><article class="impact-kpi negative"><span>现场孪生保真度</span><strong>待接入港口</strong><small>需授权设备图谱与实现结果</small></article><article class="impact-kpi negative"><span>预测区间校准</span><strong>待接入港口</strong><small>需独立现场校准窗口</small></article><article class="impact-kpi negative"><span>网安 / HIL 验收</span><strong>待接入港口</strong><small>需授权适配器、PLC/BMS 与故障注入</small></article><article class="impact-kpi neutral"><span>生产控制权</span><strong>关闭</strong><small>建议态 · 现场验收前不下发</small></article>`;
      $('impactBoundary').textContent=payload.claim_boundary;
    }).catch(()=>{
      if(generation!==impactRenderGeneration) return;
      $('impactHeadline').textContent='孪生可靠性后端未就绪'; metrics.innerHTML='<div class="empty-state">后端检查失败，系统未生成伪造通过数据。</div>';
      $('impactBoundary').textContent='请查看 /api/v3/twin/reliability 与服务日志。';
    });
    return;
  }
  if(view==='deployment'){
    $('impactStatus').textContent='FAIL-CLOSED PRODUCTION READINESS'; $('impactHeadline').textContent='正在执行部署自检';
    $('impactDescription').textContent='生产准入不再只检查“配置了文件路径”；现场图谱、实测校准和影子验收必须通过内容校验、哈希留痕且 site_id 一致。';
    metrics.innerHTML='<div class="skeleton wide"></div>';
    fetch('/health/ready',{cache:'no-store'}).then(response=>response.json()).then(payload=>{
      if(generation!==impactRenderGeneration) return;
      const labels={canonical_dataset:'公开基准数据',rl_runtime:'RL 运行时',cors:'CORS 白名单',api_authentication:'API 身份认证',privileged_api_key:'管理员密钥',api_rate_limit:'API 限流',request_body_limit:'请求体上限',security_headers:'安全响应头',production_mode:'生产模式',tls_termination:'TLS 终止声明',secret_manager:'密钥管理声明',twin_graph:'授权孪生图谱',site_calibration:'实测校准证据',shadow_acceptance:'影子运行验收',site_evidence_consistency:'现场证据一致性'};
      const siteChecks=new Set(['twin_graph','site_calibration','shadow_acceptance','site_evidence_consistency']);
      const developmentDeferred=new Set(['cors','api_authentication','privileged_api_key']);
      const productionMode=payload.checks?.production_mode?.ok===true;
      $('impactStatus').textContent=payload.production_site_ready?'PRODUCTION SITE READY':'PRODUCTION ADMISSION CLOSED';
      $('impactHeadline').textContent=payload.production_site_ready?'现场生产准入检查通过':'开源运行可用 · 生产准入未通过';
      $('impactDescription').textContent=`开源运行 ${payload.open_source_runtime_ready?'已就绪':'未就绪'}；生产现场 ${payload.production_site_ready?'已准入':'仍失效关闭'}。红色项是真实的现场替换和安全验收缺口。`;
      metrics.innerHTML=Object.entries(payload.checks||{}).map(([name,row])=>{const deferred=!productionMode&&developmentDeferred.has(name);const state=row.ok?(deferred?'开发态免要求':'通过'):siteChecks.has(name)?'待接入港口':'未配置';return `<article class="impact-kpi ${row.ok?(deferred?'neutral':'positive'):'negative'}"><span>${escapeHTML(labels[name]||name)}</span><strong>${state}</strong><small>${escapeHTML(row.status||row.mode||row.requirement||row.attestation||'失效安全门禁')}</small></article>`}).join('');
      $('impactBoundary').textContent=payload.boundary;
    }).catch(()=>{
      if(generation!==impactRenderGeneration) return;
      $('impactHeadline').textContent='部署自检接口不可用'; metrics.innerHTML='<div class="empty-state">未读取到就绪报告，生产准入保持关闭。</div>';
    });
    return;
  }
  if(view==='rl'){
    const selected=advantage?.selected;
    if(!selected){ metrics.innerHTML='<div class="empty-state">RL 正式证据尚未生成</div>'; return; }
    $('impactStatus').textContent=`${advantage.claim_status?.replaceAll('_',' ')||'EVIDENCE'} / 3 SEEDS`; $('impactHeadline').textContent=`${selected.name} 综合优势 ${pct(selected.weighted_relative_improvement?.mean)}`;
    $('impactDescription').textContent='算法只由验证集选出，盲测只做最终报告；成本、碳和峰值如为负值会原样展示。';
    metrics.innerHTML=Object.entries(selected.metrics_relative_to_fcfs||{}).map(([name,row])=>`<article class="impact-kpi ${Number(row.mean)<0?'negative':'positive'}"><span>${escapeHTML(metricNames[name]||name)}</span><strong>${pct(row.mean)}</strong><small>95% CI ${pct(row.ci_low)} ～ ${pct(row.ci_high)}</small></article>`).join('');
    $('impactBoundary').textContent=advantage.claim_boundary;
    return;
  }
  if(view==='outcome'){
    const selected=advantage?.selected; const rows=selected?.blind_test_metrics||{};
    if(!selected || !Object.keys(rows).length){ metrics.innerHTML='<div class="empty-state">绝对业务结果将在正式 V3 证据导出后显示</div>'; return; }
    const names=['throughput_teu','service_completion_ratio','delay_index_mean','queue_peak_teu','queue_end_teu','cost_per_teu','carbon_kg_per_teu','energy_kwh_per_teu','peak_kw','grid_energy_kwh'];
    $('impactStatus').textContent='BLIND TEST ABSOLUTE OUTCOMES / 3 SEEDS'; $('impactHeadline').textContent=`${selected.name} · 10 组盲测窗口的业务结果`;
    $('impactDescription').textContent='显示策略在未参与训练的末段时间窗上的绝对结果；每一项同时给出跨随机种子的 95% bootstrap 区间。';
    metrics.innerHTML=names.filter(name=>rows[name]).map(name=>{const row=rows[name];return `<article class="impact-kpi neutral"><span>${escapeHTML(metricNames[name]||name)}</span><strong>${metricValue(name,row.mean)}</strong><small>${escapeHTML(metricUnits[name]||'')} · 95% CI ${metricValue(name,row.ci_low)}～${metricValue(name,row.ci_high)}</small></article>`}).join('');
    $('impactBoundary').textContent='绝对结果来自公开数据离线环境；用于比较与现场验收设计，不冒充码头实测 KPI。';
    return;
  }
  if(view==='safety'){
    const selected=advantage?.selected; const rows=selected?.blind_test_metrics||{}; const admission=selected?.safety_admission||{};
    if(!selected || !Object.keys(rows).length){ metrics.innerHTML='<div class="empty-state">安全稳健性指标将在正式 V3 证据导出后显示</div>'; return; }
    const names=['guardrail_violation_rate','action_projection_rate','action_projection_correction_kw_mean','action_projection_severity_mean','action_projection_terminal_reachability_rate','action_projection_soc_bound_rate','action_projection_grid_cap_rate','action_projection_power_bound_rate','terminal_soc_error'];
    const projection=advantage?.projection_hardening||{}; const threshold=Number(advantage?.benchmark_contract?.eligibility?.action_projection_rate_max);
    $('impactStatus').textContent='SAFETY ADMISSION / FAIL CLOSED'; $('impactHeadline').textContent=`${selected.name} · 硬约束与策略依赖度审计`;
    $('impactDescription').textContent=`准入上限：违规率 0%，动作投影率 ≤ ${Number.isFinite(threshold)?(threshold*100).toFixed(0):'—'}%，期末 SOC 误差 ≤ 10⁻⁶；原始动作修正原因和幅度全部留痕。`;
    metrics.innerHTML=(projection.current_mean!=null?`<article class="impact-kpi positive"><span>V3.1 → V3.2 投影依赖</span><strong>${metricValue('action_projection_rate',projection.historical_mean)} → ${metricValue('action_projection_rate',projection.current_mean)}</strong><small>相对下降 ${metricValue('action_projection_rate',projection.relative_reduction)} · 历史报告已归档</small></article>`:'')+names.filter(name=>rows[name]).map(name=>{const row=rows[name];const warning=name==='action_projection_rate'&&Number(row.mean)>threshold;return `<article class="impact-kpi ${warning?'negative':'neutral'}"><span>${escapeHTML(metricNames[name]||name)}</span><strong>${metricValue(name,row.mean)}</strong><small>${escapeHTML(metricUnits[name]||'')} · 95% CI ${metricValue(name,row.ci_low)}～${metricValue(name,row.ci_high)}</small></article>`}).join('')+`<article class="impact-kpi ${admission.passed?'positive':'negative'}"><span>候选准入结果</span><strong>${admission.passed?'通过':'拒绝'}</strong><small>三种子最大投影 ${metricValue('action_projection_rate',admission.action_projection_rate_max_observed)}</small></article>`;
    $('impactBoundary').textContent='动作投影率会原样暴露策略对安全层的依赖；高投影即使无违规，也不能被包装成成熟现场策略。';
    return;
  }
  if(view==='strong'){
    const evidence=overviewData.strong_baselines; const comparisons=evidence?.comparisons||{}; const gate=evidence?.strong_baseline_gate||{};
    if(!evidence || !Object.keys(comparisons).length){ metrics.innerHTML='<div class="empty-state">强基线同窗证据尚未生成</div>'; return; }
    const labels={fcfs_neutral:'FCFS 中性基线',engineering_ops_rule:'工程作业规则代理',mpc:'MPC 滚动优化'};
    $('impactStatus').textContent='PAIRED BLIND WINDOWS / STRONG BASELINE GATE';
    $('impactHeadline').textContent=gate.all_comparators_strictly_beaten?'SAC 已严格击败全部公开基线':'SAC 未严格击败全部强基线';
    $('impactDescription').textContent='同一 10 个盲测窗口比较三种子 SAC 集成与 FCFS、工程规则代理、MPC；负值和未通过结果原样显示。工程规则不是上海现场实测 incumbent。';
    metrics.innerHTML=Object.entries(comparisons).map(([name,row])=>{const score=row.weighted_relative_improvement||{};return `<article class="impact-kpi ${row.strict_advantage_95ci?'positive':'negative'}"><span>${escapeHTML(labels[name]||name)}</span><strong>${pct(score.mean)}</strong><small>95% CI ${pct(score.ci_low)}～${pct(score.ci_high)} · ${row.strict_advantage_95ci?'严格优势成立':'未证明严格优势'}</small></article>`}).join('')+`<article class="impact-kpi ${gate.measured_current_operations_baseline_available?'positive':'negative'}"><span>现场 incumbent 日志</span><strong>${gate.measured_current_operations_baseline_available?'已接入':'待接入港口'}</strong><small>需时间戳、SOP 模式、原始动作和实际执行结果</small></article><article class="impact-kpi ${gate.production_claim_admitted?'positive':'negative'}"><span>生产收益准入</span><strong>${gate.production_claim_admitted?'通过':'关闭'}</strong><small>FCFS 单基线不足以授权集团收益声明</small></article>`;
    $('impactBoundary').textContent=evidence.claim_boundary;
    return;
  }
  if(view==='mpc'){
    if(!impact){ metrics.innerHTML='<div class="empty-state">MPC 价值情景尚未生成</div>'; return; }
    const value=impact.scenario_value||{}; const rows=impact.metrics||{}; const efficiency=impact.mpc_efficiency_value||{};
    $('impactStatus').textContent='PAIRED BLIND WINDOWS / DESCRIPTIVE'; $('impactHeadline').textContent=`MPC 吞吐 ${pct(rows.throughput_teu?.relative_improvement)} · 单位成本改善 ${pct(efficiency.cost_per_teu_relative_improvement)}`;
    $('impactDescription').textContent=`MPC 与 FCFS 在 ${impact.comparison?.window_count||'—'} 个相同盲测窗口配对；总电费/总碳会随作业量增加，等量吞吐价值单独核算。`;
    metrics.innerHTML=['throughput_teu','delay_index_mean','energy_cost','carbon_kg','peak_kw'].map(name=>{const row=rows[name]||{};const positive=Number(row.absolute_improvement)>=0;return `<article class="impact-kpi ${positive?'positive':'negative'}"><span>${escapeHTML(metricNames[name]||name)}</span><strong>${pct(row.relative_improvement)}</strong><small>${metricValue(name,row.absolute_improvement)} ${escapeHTML(metricUnits[name]||'')} · 配对 CI ${metricValue(name,row.paired_window_improvement?.ci_low)}～${metricValue(name,row.paired_window_improvement?.ci_high)}</small></article>`}).join('')+`<article class="impact-kpi ${Number(efficiency.cost_per_teu_relative_improvement)>=0?'positive':'negative'}"><span>MPC 单位吞吐成本</span><strong>${pct(efficiency.cost_per_teu_relative_improvement)}</strong><small>等量吞吐年化避免 ¥${fmt.format(Math.round(efficiency.annualized_avoided_cost||0))}</small></article><article class="impact-kpi ${Number(efficiency.carbon_per_teu_relative_improvement)>=0?'positive':'negative'}"><span>MPC 单位吞吐碳强度</span><strong>${pct(efficiency.carbon_per_teu_relative_improvement)}</strong><small>等量吞吐年化避免 ${(Number(efficiency.annualized_avoided_carbon_kg||0)/1000).toFixed(2)} tCO₂</small></article>`;
    $('impactBoundary').textContent=impact.claim_boundary;
    return;
  }
  if(view==='unit'){
    const value=impact?.learned_efficiency_value;
    if(!value){ metrics.innerHTML='<div class="empty-state">等量吞吐价值证据尚未生成</div>'; return; }
    const cost=value.cost_per_teu_relative_improvement||{}; const carbon=value.carbon_per_teu_relative_improvement||{};
    $('impactStatus').textContent=value.claim_status?.replaceAll('_',' ')||'EVIDENCE PENDING'; $('impactHeadline').textContent=`${value.name} 等量吞吐年化避免成本 ¥${fmt.format(Math.round(value.annualized_avoided_cost||0))}`;
    $('impactDescription').textContent='用 FCFS 单位成本/单位碳强度处理相同吞吐所需的资源，与学习策略比较；它衡量单位效率价值，不等于电费账单直接下降。';
    metrics.innerHTML=`<article class="impact-kpi ${Number(cost.mean)>=0?'positive':'negative'}"><span>单位吞吐成本</span><strong>${pct(cost.mean)}</strong><small>95% CI ${pct(cost.ci_low)} ～ ${pct(cost.ci_high)}</small></article><article class="impact-kpi ${Number(carbon.mean)>=0?'positive':'negative'}"><span>单位吞吐碳强度</span><strong>${pct(carbon.mean)}</strong><small>95% CI ${pct(carbon.ci_low)} ～ ${pct(carbon.ci_high)}</small></article><article class="impact-kpi neutral"><span>年化避免成本情景</span><strong>¥${fmt.format(Math.round(value.annualized_avoided_cost||0))}</strong><small>48 小时结果机械外推</small></article><article class="impact-kpi neutral"><span>年化避免碳情景</span><strong>${(Number(value.annualized_avoided_carbon_kg||0)/1000).toFixed(2)} t</strong><small>等量吞吐，不是碳核证</small></article><article class="impact-kpi positive"><span>生产控制权</span><strong>关闭</strong><small>仅公开数据离线建议态</small></article>`;
    $('impactBoundary').textContent=`${impact.claim_boundary} 等量吞吐避免值不是绝对账单节省。`;
    return;
  }
  $('impactStatus').textContent='REPRODUCIBLE EVIDENCE CONTRACT'; $('impactHeadline').textContent='70 / 10 / 20 时间隔离 · 训练不渲染 · 盲测才回放';
  $('impactDescription').textContent='模型比较绑定数据哈希、环境版本、动作维度、随机种子、训练步数和安全门；历史 V1/V2 证据保持追加式留痕。';
  metrics.innerHTML=`<article class="impact-kpi neutral"><span>数据 SHA-256</span><strong class="hash-value">${escapeHTML(overviewData.dataset.sha256.slice(0,16))}…</strong><small>完整值可从机器证据 API 查看</small></article><article class="impact-kpi neutral"><span>Train</span><strong>${fmt.format(overviewData.dataset.train_rows)}</strong><small>只参与参数优化</small></article><article class="impact-kpi neutral"><span>Validation</span><strong>${fmt.format(overviewData.dataset.validation_rows)}</strong><small>只参与选择与调参</small></article><article class="impact-kpi neutral"><span>Blind Test</span><strong>${fmt.format(overviewData.dataset.test_rows)}</strong><small>训练阶段不可见</small></article><article class="impact-kpi positive"><span>生产控制权</span><strong>关闭</strong><small>现场验收前仅建议态</small></article>`;
  $('impactBoundary').textContent='所有金额、碳排和效率数字必须同时携带数据来源、协议与现场替换边界。';
}

function openDrawer({kicker,title,lead,body}){
  $('detailKicker').textContent=kicker; $('detailTitle').textContent=title; $('detailLead').textContent=lead; $('detailBody').innerHTML=body;
  $('detailBackdrop').hidden=false; document.body.classList.add('drawer-open'); $('detailClose').focus();
}
function closeDrawer(){ $('detailBackdrop').hidden=true; document.body.classList.remove('drawer-open'); }
function trainingTrace(trace){
  const points=trace?.points||[]; if(points.length<2) return '';
  if(trace.reward_available===false){
    const first=points[0]; const last=points.at(-1); const progress=Math.max(0,Math.min(100,Number(last.progress||0)*100));
    return `<section class="training-trace progress-only"><div><span>TRAINING PROGRESS / ${fmt.format(trace.observed_points)} CALLBACK CHECKPOINTS</span><b>${fmt.format(first.step||0)} → ${fmt.format(last.step||0)} steps</b></div><div class="trace-progress-track" role="img" aria-label="真实训练进度 ${progress.toFixed(1)}%"><i style="width:${progress.toFixed(2)}%"></i></div><small>该后端回调仅持久化 progress/step，未暴露 reward/loss；页面保留真实进度证据，不伪造收敛曲线。完整末态 ${escapeHTML(JSON.stringify(trace.final_optimizer_snapshot||{}))}</small></section>`;
  }
  const values=points.map(row=>Number(row.reward_mean)); const min=Math.min(...values); const max=Math.max(...values); const span=Math.max(1e-9,max-min);
  const polyline=points.map((row,index)=>`${(index/(points.length-1)*100).toFixed(2)},${(38-(Number(row.reward_mean)-min)/span*34).toFixed(2)}`).join(' ');
  return `<section class="training-trace"><div><span>TRAINING TRACE / ${fmt.format(trace.observed_points)} CHECKPOINTS</span><b>reward ${safeNumber(values[0],2)} → ${safeNumber(values.at(-1),2)}</b></div><svg viewBox="0 0 100 42" preserveAspectRatio="none" role="img" aria-label="真实训练奖励轨迹"><path d="M0 38H100"/><polyline points="${polyline}"/></svg><small>抽样展示持久化 metrics.jsonl；完整 optimizer 末态 ${escapeHTML(JSON.stringify(trace.final_optimizer_snapshot||{}))}</small></section>`;
}

async function openAlgorithmEvidence(algorithmId){
  const response=await fetch(`/api/v3/algorithms/${encodeURIComponent(algorithmId)}/evidence`,{cache:'no-store'}); if(!response.ok) throw new Error(`algorithm evidence ${response.status}`);
  const row=await response.json();
  const historicalRows=(row.historical_evidence?.runs||[]).map(run=>`<tr><td>${escapeHTML(run.dataset_id)}</td><td>${escapeHTML(run.environment_version||'v1')}</td><td>${escapeHTML(run.seed??'controller')}</td><td>${fmt.format(run.total_steps||0)}</td><td><code>${escapeHTML(run.job_id)}</code></td><td>${metricValue('throughput_teu',run.metrics?.throughput_teu)}</td><td>${metricValue('delay_index_mean',run.metrics?.delay_index_mean)}</td><td>${metricValue('energy_cost',run.metrics?.energy_cost)}</td><td>${metricValue('carbon_kg',run.metrics?.carbon_kg)}</td></tr>`).join('');
  const historical=`<details class="history-ledger"><summary>历史正式指标留痕 · ${fmt.format(row.historical_evidence?.runs?.length||0)} RUNS</summary><div><table class="metric-table"><thead><tr><th>数据集</th><th>环境</th><th>Seed</th><th>步数</th><th>Job ID</th><th>吞吐</th><th>延误</th><th>成本</th><th>碳</th></tr></thead><tbody>${historicalRows||'<tr><td colspan="9">暂无历史正式记录</td></tr>'}</tbody></table></div></details>`;
  const profiles=(row.v3_profiles||[]).map(profile=>{
    const trace=row.training_traces?.[profile.job_ids?.at(-1)];
    const metricRows=Object.entries(profile.metrics||{}).filter(([name])=>metricNames[name]).map(([name,summary])=>`<tr><td>${escapeHTML(metricNames[name]||name)}</td><td>${metricValue(name,summary.mean)} ${escapeHTML(metricUnits[name]||'')}</td><td>${metricValue(name,summary.ci_low)} ～ ${metricValue(name,summary.ci_high)}</td></tr>`).join('');
    return `<section class="detail-section"><div class="detail-section-head"><h3>${profile.id==='default_port_profile'?'默认港口目标':escapeHTML(profile.id)}</h3><b>${profile.formal_runs} RUNS · SEEDS ${escapeHTML(profile.seeds.join(', '))}</b></div><div class="detail-chips"><span>${row.trainable?fmt.format(profile.minimum_optimizer_steps)+' optimizer steps':'确定性控制器'}</span><span>10 blind episodes</span><span>训练渲染 ${profile.render_calls_during_training===0?'0 次':'异常'}</span></div>${trainingTrace(trace)}<table class="metric-table"><thead><tr><th>指标</th><th>均值</th><th>95% bootstrap CI</th></tr></thead><tbody>${metricRows||'<tr><td colspan="3">暂无正式指标</td></tr>'}</tbody></table><details><summary>作业、奖励与模型哈希</summary><pre>${escapeHTML(JSON.stringify({job_ids:profile.job_ids,model_sha256:profile.model_sha256,reward_weights:profile.reward_weights},null,2))}</pre></details></section>`;
  }).join('');
  openDrawer({kicker:`${row.family.toUpperCase()} / ${row.id.toUpperCase()}`,title:row.name,lead:`${row.description} · ${row.implementation}`,body:`<div class="detail-facts"><span>动作空间<b>${row.action_space==='continuous'?'连续':'离散'}</b></span><span>V3 正式运行<b>${row.formal_runs}</b></span><span>历史正式运行<b>${row.historical_formal_runs}</b></span><span>训练渲染<b>关闭</b></span></div>${profiles||'<div class="detail-empty">尚无满足正式门禁的 V3 训练结果。</div>'}${historical}<p class="detail-boundary">${escapeHTML(row.claim_boundary)}</p>`});
}

function openCapabilityDetail(capabilityId){
  const row=(overviewData?.capabilities||[]).find(item=>item.id===capabilityId); if(!row) return;
  const depth=row.depth||{};
  const list=(title,items)=>`<section class="detail-list"><h3>${title}</h3><ul>${(items||[]).map(item=>`<li>${escapeHTML(item)}</li>`).join('')}</ul></section>`;
  const artifactRows=(depth.code_artifacts||[]).map(item=>`${item.path} · ${item.exists?'SHA-256 '+item.sha256:'缺失'}`);
  const execution=[`执行等级：${depth.implementation_label||depth.implementation_level}`,`决策来源：${depth.decision_source}`,`当前数据：${depth.current_data_mode}`,`独立模型输出：${depth.model_output_available?'有':'无'}`,`生产准入：${depth.production_ready?'通过':'关闭'}`,`失效回退：${depth.fail_closed_fallback}`];
  openDrawer({kicker:`BUSINESS DOMAIN / ${row.id.toUpperCase()}`,title:row.name,lead:`${depth.implementation_label||stateNames[row.state]} · ${row.engine}`,body:`<div class="detail-grid">${list('执行状态与真实输出来源',execution)}${list('可调用运行接口',depth.runtime_endpoints)}${list('状态输入',depth.state_inputs)}${list('决策输出',depth.decision_outputs)}${list('硬约束与失效安全',depth.hard_constraints)}${list('训练后 / 现场验收指标',depth.acceptance_metrics)}${list('代码与 SHA-256 证据',artifactRows)}${list('阻止现场准入的缺口',depth.site_blockers)}</div><section class="site-replace"><span>SITE DATA REPLACEMENT</span><p>${escapeHTML(row.site_replacement)}</p></section><p class="detail-boundary">公开数据阶段仅验证软件、仿真、监测或离线策略链；无独立优化器的域明确标注，现场字段缺失时不产生生产控制权。</p>`});
}
function openGateDetail(gateId){
  const row=(overviewData?.deployment_gates||[]).find(item=>item.id===gateId); if(!row) return;
  const list=(title,items)=>`<section class="detail-list"><h3>${title}</h3><ul>${(items||[]).map(item=>`<li>${escapeHTML(item)}</li>`).join('')}</ul></section>`;
  openDrawer({kicker:`DEPLOYMENT GATE / ${row.id.toUpperCase()}`,title:row.name,lead:row.state==='software_ready'?'软件能力已具备，现场证据仍需审批。':'现场部署前必须完成，未通过即失效安全。',body:`<div class="detail-grid">${list('所需证据',row.required_evidence)}${list('通过条件',row.pass_criteria)}</div><section class="site-replace"><span>FAIL-CLOSED ACTION</span><p>${escapeHTML(row.failure_action)}</p></section>`});
}
function openPortLineage(datasetId){
  const row=(readinessData?.ports||[]).find(item=>item.dataset_id===datasetId); if(!row) return;
  const chips=(title,items)=>`<section class="detail-list"><h3>${title}</h3><ul>${(items||[]).map(item=>`<li>${escapeHTML(item)}</li>`).join('')||'<li>无</li>'}</ul></section>`;
  const sources=(row.sources||[]).map(source=>`<li><b>${escapeHTML(source.publisher||'公开来源')}</b>${source.url?`<a href="${escapeHTML(source.url)}" target="_blank" rel="noreferrer">打开原始公开链接 ↗</a>`:'<span>来源地址见数据卡</span>'}</li>`).join('');
  openDrawer({kicker:`DATA LINEAGE / ${row.dataset_id.toUpperCase()}`,title:row.dataset_id,lead:`${fmt.format(row.rows)} rows · ${fmt.format(row.independent_source_observations)} independent source observations · ${row.recommended_role}`,body:`<div class="detail-facts"><span>证据等级<b>${escapeHTML(row.evidence_tier)}</b></span><span>数据 SHA-256<b class="wrap-hash">${escapeHTML(row.dataset_sha256)}</b></span><span>失效策略<b>${readinessData.fail_closed?'Fail closed':'—'}</b></span><span>生产遥测<b>否</b></span></div><div class="detail-grid">${chips('公开实测 / 官方字段',row.measured_columns)}${chips('工程派生字段',row.derived_columns)}${chips('仍不可用字段',row.unavailable_factors)}<section class="detail-list source-list"><h3>来源与原始链接</h3><ul>${sources}</ul></section></div><p class="detail-boundary">数据行数不等于独立信息量；现场 KPI 只能在授权字段替换、校准和影子运行通过后声明。</p>`});
}

async function loadOverview(){
  const response=await fetch('/api/v3/overview',{cache:'no-store'}); if(!response.ok) throw new Error(`overview ${response.status}`);
  const data=await response.json();
  overviewData=data;
  $('heroVersion').textContent=data.version;
  $('algorithmCount').textContent=data.algorithms.length;
  $('datasetRows').textContent=fmt.format(data.dataset.rows);
  $('sourceObservations').textContent=fmt.format(data.dataset.independent_source_observations);
  $('formalRuns').textContent=fmt.format(data.evidence.formal_run_count);
  $('generatedAt').textContent=`EVIDENCE UPDATED ${new Date(data.generated_at).toLocaleString('zh-CN',{hour12:false})}`;
  renderAlgorithms(data.algorithms); renderCapabilities(data.capabilities); renderGates(data.deployment_gates); renderAdvantage(data.advantage); renderImpact(document.querySelector('[data-impact-view].active')?.dataset.impactView||'rl');
}

async function loadReadiness(){
  const response=await fetch('/api/v3/data-readiness',{cache:'no-store'}); if(!response.ok) throw new Error(`readiness ${response.status}`);
  const data=await response.json();
  readinessData=data;
  $('replacementList').innerHTML=data.mandatory_site_replacements.map(item=>`<span>${item}</span>`).join('');
}

function liveValue(obj,key){ const n=Number(obj?.[key]); return Number.isFinite(n)?n:null; }
function renderPublicConditions(values,{label,color,currentUnit='m/s'}={}){
  $('liveState').textContent=label; $('liveState').style.color=color;
  $('temperature').textContent=safeNumber(values.temperature,1);
  $('wind').textContent=safeNumber(values.wind,1);
  $('wave').textContent=safeNumber(values.wave,2);
  $('current').textContent=safeNumber(values.current,2);
  $('temperatureUnit').textContent='°C'; $('windUnit').textContent='m/s';
  $('waveUnit').textContent='m'; $('currentUnit').textContent=currentUnit;
}
async function loadLive(){
  try{
    const response=await fetch('/api/v3/public-data/shanghai/live',{cache:'no-store'});
    if(!response.ok) throw new Error(`PUBLIC FEED ${response.status}`);
    const data=await response.json();
    renderPublicConditions({temperature:liveValue(data.weather,'temperature_2m'),wind:liveValue(data.weather,'wind_speed_10m'),wave:liveValue(data.marine,'wave_height'),current:liveValue(data.marine,'ocean_current_velocity')},{label:'公开模型源已连接 · 非现场遥测',color:'#6fe4cf',currentUnit:data.marine_units?.ocean_current_velocity||'km/h'});
    $('temperatureUnit').textContent=data.weather_units?.temperature_2m||'°C'; $('windUnit').textContent=data.weather_units?.wind_speed_10m||'m/s'; $('waveUnit').textContent=data.marine_units?.wave_height||'m';
  }catch(error){
    try{
      const response=await fetch('/api/v3/runtime/frame',{cache:'no-store'}); if(!response.ok) throw new Error(`RUNTIME ${response.status}`);
      const data=await response.json(); const state=data.public_conditions||{};
      if(!data.available || !Number.isFinite(Number(state.ambient_c))) throw new Error('calibrated public conditions unavailable');
      renderPublicConditions({temperature:state.ambient_c,wind:state.wind_speed_mps,wave:state.wave_height_m,current:state.current_speed_mps},{label:'公开校准连续回放 · 连续非实测',color:'#f2b45c',currentUnit:'m/s'});
      toast('外部公开源暂不可达；当前显示仓库内哈希固定公开快照的连续校准回放。');
    }catch(runtimeError){
      $('liveState').textContent='公开数据链不可用 · 已失效安全'; $('liveState').style.color='#ee786f'; toast('公开数据与校准回放均不可用；系统未填充伪数据。');
    }
  }
}

function restoreHashTarget(){
  const id=decodeURIComponent(window.location.hash.replace(/^#/,''));
  if(!id) return;
  const target=document.getElementById(id);
  if(target) target.scrollIntoView({block:'start',behavior:'auto'});
}

Promise.allSettled([loadOverview(),loadReadiness(),loadLive()]).then(results=>{
  const failed=results.filter(item=>item.status==='rejected');
  if(failed.length) toast(`有 ${failed.length} 个事实接口未加载；系统未填充伪数据。`);
  window.setTimeout(restoreHashTarget,80);
});
window.addEventListener('hashchange',restoreHashTarget);
setInterval(()=>loadOverview().catch(()=>{}),30000);
setInterval(()=>loadLive().catch(()=>{}),60000);

document.addEventListener('click',event=>{
  const impactButton=event.target.closest('[data-impact-view]'); if(impactButton){ renderImpact(impactButton.dataset.impactView); return; }
  const algorithmButton=event.target.closest('[data-algorithm]'); if(algorithmButton){ openAlgorithmEvidence(algorithmButton.dataset.algorithm).catch(()=>toast('算法证据读取失败；未填充模拟指标。')); return; }
  const capabilityButton=event.target.closest('[data-capability]'); if(capabilityButton){ openCapabilityDetail(capabilityButton.dataset.capability); return; }
  const gateButton=event.target.closest('[data-gate]'); if(gateButton){ openGateDetail(gateButton.dataset.gate); return; }
  const portButton=event.target.closest('[data-port]'); if(portButton){ openPortLineage(portButton.dataset.port); return; }
  if(event.target===$('detailBackdrop')) closeDrawer();
});
$('detailClose').addEventListener('click',closeDrawer);
document.addEventListener('keydown',event=>{ if(event.key==='Escape'&&!$('detailBackdrop').hidden) closeDrawer(); });
