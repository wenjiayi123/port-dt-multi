const bars = [
  ['靠泊窗口', 78], ['堆场压力', 64], ['岸电负荷', 71], ['AGV 调度', 83], ['碳强度', 52]
];
let strategies = [
  {name:'保守稳态策略', tag:'SAFE', save:'-6.8%', risk:'+0.03', trust:'0.91'},
  {name:'平衡收益策略', tag:'RECOMMENDED', save:'-10.5%', risk:'+0.08', trust:'0.87'},
  {name:'激进削峰策略', tag:'REVIEW', save:'-14.2%', risk:'+0.18', trust:'0.73'}
];
const logs = [
  '[Twin] 读取港区状态：靠泊高峰窗口 T+30/T+60 已识别',
  '[Berth] ETA 集中度 76，泊位窗口占用 82，岸桥资源集中 69',
  '[AGV] 任务队列强度 86，换电/补能排队 81，路径冲突压力 79',
  '[BESS] SoC 当前约 61%，安全带 35%–85%，岸电覆盖比 1.14',
  '[Energy] 峰值负荷基线 3.80MW，平衡策略预计压降至 3.40MW',
  '[Yard] 堆场压力 64：箱位占用 68，集卡等待 61，翻箱强度 58',
  '[RL] 生成 3 条策略候选：保守 / 平衡 / 激进',
  '[Guardrail] SoC、服务水平、岸电保障约束检查通过',
  '[Counterfactual] 平衡策略削峰 10.5%，风险增量 +0.08',
  '[Carbon] 60 分钟窗口预计减排约 0.51 吨 CO2e，约 7.4%',
  '[Cost] 分时电价错峰后，预计 60 分钟电费下降约 8.1%',
  '[HITL] 建议进入 dry-run，由人工确认后写入审计链',
  '[Audit] 已生成策略版本、输入快照、护栏结果与回执节点',
  '[Loop] 下一轮滚动刷新：持续监听靠泊、AGV、岸电、堆场与碳能雷达'
];

const shanghaiCalcNotes = {
  '靠泊窗口': {
    title: '靠泊窗口 · 78 / 上海港高峰靠泊压力',
    body: '原信息：该指标用于判断港口未来 30–60 分钟压力变化，数值越高越需要关注泊位、岸桥与靠泊资源边界。\n计算口径：78 = 0.50×泊位窗口占用82 + 0.30×ETA集中度76 + 0.20×岸桥资源集中69。\n上海港解释：适配洋山深水港/外高桥这类高吞吐集装箱港场景；当大型集装箱船靠泊窗口集中时，岸桥、AGV、堆场和岸电需求会同步抬升。\n使用方式：>75 进入重点盯控；>85 建议提前锁定岸桥与堆场资源。'
  },
  '堆场压力': {
    title: '堆场压力 · 64 / 箱区作业压力',
    body: '原信息：该指标用于判断堆场资源边界、调度拥堵和执行风险。\n计算口径：64 = 0.45×箱位占用压力68 + 0.35×闸口/集卡等待61 + 0.20×翻箱强度58。\n上海港解释：上海港箱量大、集疏运节奏强，堆场压力不是只看堆存率，还要看翻箱、闸口到达和作业波峰。\n使用方式：当前为中等偏上，适合执行温和优化，不适合叠加激进压缩策略。'
  },
  '岸电负荷': {
    title: '岸电负荷 · 71 / 靠港船舶岸电保供',
    body: '原信息：该指标用于判断岸电保供、储能 SoC 和峰值负荷风险。\n计算口径：71 = 0.50×岸电实时负荷率74 + 0.30×BESS保供压力69 + 0.20×同时在泊受电需求67。\n上海港解释：上海港持续推进靠港船舶使用岸电，因此能源策略必须先保证船舶受电安全，再考虑削峰降本。\n使用方式：>70 表示岸电进入重点保障区，BESS 不能被过度用于经济套利。'
  },
  'AGV 调度': {
    title: 'AGV 调度 · 83 / 自动化运输链压力',
    body: '原信息：该指标用于判断 AGV 调度、拥堵、充换电队列和任务延迟。\n计算口径：83 = 0.45×任务队列强度86 + 0.35×换电/补能排队81 + 0.20×路径冲突压力79。\n上海港解释：适配洋山四期自动化码头 AGV 场景；靠泊高峰会先传导到 AGV 运力和换电节奏。\n使用方式：这是当前最高压力项，应优先观察是否需要充电后移、路径重分配或换电区保护。'
  },
  '碳强度': {
    title: '碳强度 · 52 / 低碳运行压力',
    body: '原信息：该指标用于判断碳排压力和低碳调度收益。\n计算口径：52 = 0.60×电网碳因子指数56 + 0.25×柴油替代残留49 + 0.15×清洁电力吸纳不足46。\n上海港解释：上海港自动化、电动化和岸电使用提升后，碳压力可低于作业压力；但在高负荷窗口仍有降碳优化空间。\n使用方式：该项越低越好；当前属于可控区。'
  }
};

const strategyCalcNotes = {
  '保守稳态策略': '原信息：保守策略优先稳定生产和安全边界，适合高压窗口或模型置信不足时使用。\n计算口径：节能率 = (12.50MWh 基线 - 11.65MWh 执行后) / 12.50MWh = 6.8%。\n贡献拆解：BESS轻度削峰3.1pct + 堆场照明分区调光2.2pct + HVAC预冷/限幅1.5pct。\n风险由来：风险分 0.31 → 0.34，因此 +0.03。\n上海港解释：适合靠泊窗口紧、岸电保供优先的场景。',
  '平衡收益策略': '原信息：这是当前推荐进入 dry-run 的平衡策略，适合展示收益、风险和人工确认之间的闭环。\n计算口径：削峰率 = (3.80MW 基线峰值 - 3.40MW 执行后峰值) / 3.80MW = 10.5%。\n贡献拆解：BESS放电6.0pct + HVAC错峰2.3pct + 堆场照明1.2pct + AGV充电后移1.0pct。\n风险由来：风险分 0.31 → 0.39，因此 +0.08。\n上海港解释：符合“先保供、再削峰、再降本”的港口能源调度逻辑。',
  '激进削峰策略': '原信息：激进策略收益更高但风险抬升，需要人工复核后才能执行。\n计算口径：综合收益约 -14.2%，来自更深度 BESS 放电、AGV 充电错峰、照明压缩和 HVAC 限幅。\n风险由来：风险分 0.31 → 0.49，因此 +0.18。\n上海港解释：这种策略会压缩 SoC 保供余量和作业安全余量，只能在短时尖峰窗口由人工确认。'
};

function escAttr(s){
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function renderBars(){
  const root = document.getElementById('situationBars');
  root.innerHTML = bars.map(([name,value]) => {
    const detail = shanghaiCalcNotes[name] || { title: `${name} · 态势指标`, body: `当前 ${name} 指标为 ${value}。` };
    return `
      <div class="bar-row" data-tip-title="${escAttr(detail.title)}" data-tip-body="${escAttr(detail.body)}"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${value}%"></div></div><b>${value}</b></div>
    `;
  }).join('');
}
function renderStrategies(){
  const root = document.getElementById('strategyStack');
  root.innerHTML = strategies.map((s,i)=>`
    <div class="strategy-card ${s.active || (!strategies.some(item=>item.active) && i===1)?'active':''}" data-tip-title="${escAttr(s.name)}" data-tip-body="${escAttr(`收益 ${s.save}，风险 ${s.risk}，可信度 ${s.trust}。\n${s.note || strategyCalcNotes[s.name] || ''}`)}">
      <div class="strategy-title"><span>${s.name}</span><b>${s.tag}</b></div>
      <div class="strategy-meta"><span>收益 <b>${s.save}</b></span><span>风险 <b>${s.risk}</b></span><span>可信度 <b>${s.trust}</b></span></div>
    </div>
  `).join('');
}
function renderTimeline(){
  const root = document.getElementById('timeline');
  const timelineNotes = [
    'T+15min：到港波峰预警。由 ETA 集中度与泊位占用形成，用于提前锁定岸桥和堆场资源。',
    'T+30min：岸电叠加窗口。预计同时在泊受电需求抬升，是 BESS 与岸电保供的重点观察点。',
    'T+45min：AGV 换电高峰。任务队列与补能排队可能同步抬升，适合执行充电后移。',
    'T+60min：负荷峰值窗口。用于验证削峰策略是否把峰值从 3.80MW 压到约 3.40MW。',
    'T+75min：审批/回执检查点。用于核验策略执行结果，并写入审计回放。'
  ];
  root.innerHTML = [12,32,53,76,91].map((x,i)=>`<span class="tick" style="left:${x}%;animation-delay:${i*.22}s" data-tip-title="未来关键窗口 T+${(i+1)*15}min" data-tip-body="${escAttr('原信息：该亮点代表未来风险/机会窗口，可能是靠泊高峰、负荷峰值、储能低位、堆场压力或策略执行确认点。\n计算由来：' + timelineNotes[i])}"></span>`).join('');
}
let rlLogTicker = null;
let rlLogTimeouts = [];
let futureRunActive = false;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function writeLogs(sourceLogs = logs, repeat = true){
  const box = document.getElementById('consoleLog');
  if(!box) return;
  if(rlLogTicker) clearInterval(rlLogTicker);
  rlLogTimeouts.forEach(clearTimeout);
  rlLogTimeouts = [];
  box.innerHTML = '<div class="rl-log-stream" aria-live="polite"></div>';
  box.style.whiteSpace = 'normal';
  box.style.overflow = 'hidden';
  box.style.position = 'relative';
  const stream = box.querySelector('.rl-log-stream');
  let index = 0;
  const maxRows = 7;
  function pushLine(){
    if(!sourceLogs.length) return;
    const line = sourceLogs[index % sourceLogs.length];
    const row = document.createElement('div');
    row.className = 'rl-log-row';
    row.textContent = line;
    row.style.cssText = 'display:block; margin:0 0 8px 0; line-height:1.45; color:#bfffea; opacity:0; transform:translateY(8px); transition:opacity .28s ease, transform .28s ease; white-space:normal;';
    stream.appendChild(row);
    requestAnimationFrame(()=>{
      row.style.opacity = '1';
      row.style.transform = 'translateY(0)';
    });
    while(stream.children.length > maxRows){
      stream.removeChild(stream.firstElementChild);
    }
    index += 1;
  }
  sourceLogs.slice(0, maxRows).forEach((_, i)=> rlLogTimeouts.push(setTimeout(pushLine, i * 220)));
  if(repeat) rlLogTicker = setInterval(pushLine, 1500);
}

function setStage(stageId, state){
  const el = document.querySelector(`.run-stage[data-stage="${stageId}"]`);
  if(!el) return;
  el.classList.remove('pending','active','done','blocked');
  el.classList.add(state);
  const label = el.querySelector('em');
  if(label) label.textContent = ({pending:'WAIT',active:'RUN',done:'PASS',blocked:'BLOCK'})[state] || state;
}

function appendTerminal(message, tone = ''){
  const terminal = document.getElementById('simulationTerminal');
  if(!terminal) return;
  const idle = terminal.querySelector('.terminal-idle');
  if(idle) idle.remove();
  const line = document.createElement('div');
  line.className = `terminal-line ${tone}`.trim();
  const text = document.createElement('span');
  text.textContent = message;
  line.appendChild(text);
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

function resetRunUI(){
  ['situation','candidates','counterfactual','guardrails','receipt'].forEach(id=>setStage(id,'pending'));
  document.getElementById('runIdValue').textContent = '正在编排';
  document.getElementById('snapshotStatus').textContent = '待读取';
  document.getElementById('candidateStatus').textContent = '0 / 3';
  document.getElementById('overlayGuardStatus').textContent = '等待校验';
  document.getElementById('simulationTerminal').innerHTML = '<div class="terminal-idle">推演通道已建立，等待锁定态势……</div>';
  document.getElementById('snapshotGrid').innerHTML = [
    ['BESS 荷电状态','等待态势快照'],['岸电当前功率','等待态势快照'],['模型奖励漂移','等待可观测信号'],['策略候选池','等待候选生成']
  ].map(([name,note])=>`<div><span>${name}</span><strong>--</strong><em>${note}</em></div>`).join('');
  document.getElementById('runCandidateGrid').innerHTML = '<div class="candidate-placeholder"></div><div class="candidate-placeholder"></div><div class="candidate-placeholder"></div>';
  document.getElementById('runGuardGrid').innerHTML = '<div class="guard-placeholder"></div><div class="guard-placeholder"></div><div class="guard-placeholder"></div>';
  const decision = document.getElementById('decisionModule');
  decision.classList.remove('ready','blocked');
  document.getElementById('decisionLabel').textContent = '推演进行中';
  document.getElementById('decisionReason').textContent = '系统正在读取态势并生成反事实证据，完成前不会进入任何执行步骤。';
  document.getElementById('recommendedStrategy').textContent = '等待计算';
  document.getElementById('evidenceDigest').textContent = '等待生成';
  document.getElementById('productionBoundary').textContent = '只做候选、仿真、护栏与审计，不向生产设备下发控制指令。';
  document.getElementById('finishSimulation').disabled = true;
}

function openSimulation(){
  resetRunUI();
  const overlay = document.getElementById('simulationOverlay');
  overlay.classList.add('open');
  overlay.setAttribute('aria-hidden','false');
  document.body.classList.add('simulation-active');
  setTimeout(()=>document.getElementById('closeSimulation').focus(), 420);
}

function closeSimulation(){
  const overlay = document.getElementById('simulationOverlay');
  overlay.classList.remove('open');
  overlay.setAttribute('aria-hidden','true');
  document.body.classList.remove('simulation-active');
  document.getElementById('btnIgnite').focus();
}

function renderSnapshot(snapshot){
  const items = [
    ['BESS 荷电状态',`${Number(snapshot.bess_soc_pct || 0).toFixed(1)}%`,'MAS 态势快照'],
    ['岸电当前功率',`${Number(snapshot.shore_power_kw || 0).toFixed(0)} kW`,'岸电节点实时聚合'],
    ['模型奖励漂移',Number(snapshot.reward_drift || 0).toFixed(3),'RL Ops 可观测信号'],
    ['策略候选池',`${snapshot.candidate_pool_size || 0} 条`,'风险分层选出 3 条']
  ];
  document.getElementById('snapshotGrid').innerHTML = items.map((item,i)=>`
    <div class="revealed" style="animation-delay:${i*.08}s"><span>${item[0]}</span><strong>${item[1]}</strong><em>${item[2]}</em></div>
  `).join('');
  document.getElementById('snapshotStatus').textContent = `${snapshot.horizon_min || 90} MIN LOCKED`;
}

function renderRunCandidates(data){
  const root = document.getElementById('runCandidateGrid');
  root.innerHTML = data.candidates.map((item,i)=>{
    const isRecommended = item.id === data.recommended_strategy_id;
    const energyPct = item.baseline_energy_kwh > 0 ? item.energy_saving_kwh / item.baseline_energy_kwh * 100 : 0;
    return `<article class="run-candidate ${isRecommended?'recommended':''} ${item.dispatch_ready?'':'blocked'}" style="animation-delay:${i*.12}s">
      <div class="candidate-top"><div><span class="candidate-mode">${escAttr(item.mode)}</span><h4>${escAttr(item.title)}</h4></div><b class="candidate-tag">${isRecommended?'推荐':escAttr(item.tag)}</b></div>
      <div class="candidate-metrics"><div><span>节能</span><b>${item.energy_saving_kwh.toFixed(1)} kWh</b></div><div><span>削峰</span><b>${item.peak_reduction_kw.toFixed(1)} kW</b></div><div><span>可信度</span><b>${item.confidence.toFixed(2)}</b></div></div>
      <div class="candidate-result"><span>电耗改善 ${energyPct.toFixed(1)}%</span><strong>${item.dispatch_ready?'仿真可用':'保持阻断'}</strong></div>
    </article>`;
  }).join('');
  document.getElementById('candidateStatus').textContent = `${data.candidates.length} / ${data.candidates.length}`;
}

function renderRunGuardrails(data){
  const root = document.getElementById('runGuardGrid');
  root.innerHTML = data.guardrails.map((item,i)=>`<article class="run-guard ${item.passed?'pass':'block'}" style="animation-delay:${i*.08}s">
    <div class="guard-state"><span>${item.level === 'hard'?'HARD':'SOFT'}</span><b>${item.passed?'PASS':'BLOCK'}</b></div>
    <h4>${escAttr(item.name)}</h4>
    <div class="guard-measure">${escAttr(String(item.actual))}${escAttr(item.unit || '')}</div>
    <span class="guard-threshold">阈值 ${escAttr(item.threshold)}</span>
    <span class="guard-source">${escAttr(item.source)}</span>
  </article>`).join('');
  const hardPassed = data.guardrails.filter(item=>item.level === 'hard').every(item=>item.passed);
  document.getElementById('overlayGuardStatus').textContent = hardPassed ? 'HARD RULES PASS' : 'HARD RULE BLOCKED';
}

function renderDecision(data){
  const ready = data.decision.ready_for_human_dry_run;
  const module = document.getElementById('decisionModule');
  module.classList.remove('ready','blocked');
  module.classList.add(ready ? 'ready' : 'blocked');
  document.getElementById('decisionLabel').textContent = data.decision.label;
  document.getElementById('decisionReason').textContent = `${data.decision.next_action}。${data.decision.production_boundary}`;
  document.getElementById('recommendedStrategy').textContent = data.decision.recommended_strategy_title || '无可用策略';
  document.getElementById('evidenceDigest').textContent = data.audit.evidence_digest;
  document.getElementById('productionBoundary').textContent = data.decision.production_boundary;
}

function applyRunToMainSurface(data){
  const recommended = data.candidates.find(item=>item.id === data.recommended_strategy_id) || data.candidates[0];
  if(recommended){
    document.getElementById('trustValue').textContent = recommended.confidence.toFixed(2);
    document.getElementById('riskValue').textContent = recommended.risk_level;
  }
  strategies = data.candidates.map(item=>{
    const savingPct = item.baseline_energy_kwh > 0 ? item.energy_saving_kwh / item.baseline_energy_kwh * 100 : 0;
    return {
      name:item.title,
      tag:item.id === data.recommended_strategy_id ? 'RECOMMENDED' : item.tag,
      save:`-${savingPct.toFixed(1)}%`,
      risk:item.risk_level,
      trust:item.confidence.toFixed(2),
      active:item.id === data.recommended_strategy_id,
      note:`${item.reason}\n反事实结果：节能 ${item.energy_saving_kwh.toFixed(2)} kWh，削峰 ${item.peak_reduction_kw.toFixed(2)} kW。`
    };
  });
  renderStrategies();
  document.getElementById('counterGrid').innerHTML = data.candidates.map(item=>`
    <div class="counter-item"><span>若执行${escAttr(item.mode)}策略</span><strong>节能 ${item.energy_saving_kwh.toFixed(1)} kWh</strong><em>削峰 ${item.peak_reduction_kw.toFixed(1)} kW · ${item.dispatch_ready?'仿真可用':'护栏前阻断'}</em></div>
  `).join('');
  document.getElementById('guardList').innerHTML = data.guardrails.map(item=>`
    <li class="${item.passed?'pass':'block'}"><span></span>${escAttr(item.name)} · ${item.passed?'通过':'阻断'}</li>
  `).join('');
  const guardStatus = document.getElementById('guardStatus');
  guardStatus.textContent = data.decision.ready_for_human_dry_run ? 'SAFE' : 'BLOCKED';
  guardStatus.style.background = data.decision.ready_for_human_dry_run ? 'var(--green)' : '#ff6c7f';
  document.getElementById('aiSummary').textContent = `${data.decision.label}。推荐：${data.decision.recommended_strategy_title}。${data.decision.production_boundary}`;
  writeLogs(data.logs, false);
}

async function ignite(){
  if(futureRunActive) return;
  futureRunActive = true;
  const button = document.getElementById('btnIgnite');
  button.disabled = true;
  button.textContent = '推演中…';
  openSimulation();
  setStage('situation','active');
  appendTerminal('[Boot] 建立反事实推演通道；生产下发接口保持隔离');
  try{
    const request = fetch('/api/rl/future/run',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({horizon_min:90,step_min:5,max_candidates:3,source:'rl-future-deck'})
    });
    await sleep(560);
    const response = await request;
    const data = await response.json();
    if(!response.ok) throw new Error(data.detail || `推演接口返回 ${response.status}`);

    document.getElementById('runIdValue').textContent = data.run_id;
    renderSnapshot(data.snapshot);
    appendTerminal(data.logs[0] || '[Situation] 态势快照已锁定','pass');
    setStage('situation','done');
    setStage('candidates','active');
    await sleep(620);

    renderRunCandidates(data);
    appendTerminal(data.logs[1] || '[Candidates] 候选策略已生成','pass');
    setStage('candidates','done');
    setStage('counterfactual','active');
    await sleep(680);
    data.logs.slice(2,2 + data.candidates.length).forEach(line=>appendTerminal(line,line.includes('阻断')?'block':'pass'));
    const counterStage = data.stages.find(item=>item.id === 'counterfactual');
    setStage('counterfactual',counterStage && counterStage.status === 'blocked' ? 'blocked' : 'done');
    setStage('guardrails','active');
    await sleep(720);

    renderRunGuardrails(data);
    const guardLogStart = 2 + data.candidates.length;
    data.logs.slice(guardLogStart,guardLogStart + data.guardrails.length).forEach(line=>appendTerminal(line,line.includes('BLOCK')?'block':'pass'));
    const guardStage = data.stages.find(item=>item.id === 'guardrails');
    setStage('guardrails',guardStage && guardStage.status === 'blocked' ? 'blocked' : 'done');
    setStage('receipt','active');
    await sleep(760);

    data.logs.slice(-3).forEach((line,i)=>appendTerminal(line,i===2?'audit':(line.includes('阻断')?'block':'pass')));
    renderDecision(data);
    setStage('receipt','done');
    applyRunToMainSurface(data);
    document.getElementById('finishSimulation').disabled = false;
    button.textContent = '再次推演';
  }catch(error){
    const activeStage = document.querySelector('.run-stage.active');
    if(activeStage) setStage(activeStage.dataset.stage,'blocked');
    appendTerminal(`[Fail Closed] ${error.message}`,'block');
    appendTerminal('[Boundary] 推演未完成，未产生任何生产下发动作','audit');
    const module = document.getElementById('decisionModule');
    module.classList.add('blocked');
    document.getElementById('decisionLabel').textContent = '推演失败，保持安全阻断';
    document.getElementById('decisionReason').textContent = `${error.message}。系统未调用生产下发接口，请检查服务后重新推演。`;
    document.getElementById('runIdValue').textContent = 'FAILED-CLOSED';
    document.getElementById('finishSimulation').disabled = false;
    button.textContent = '重新推演';
  }finally{
    futureRunActive = false;
    button.disabled = false;
  }
}
renderBars();
renderStrategies();
renderTimeline();
writeLogs();
document.getElementById('btnIgnite').addEventListener('click', ignite);
document.getElementById('closeSimulation').addEventListener('click', closeSimulation);
document.getElementById('finishSimulation').addEventListener('click', closeSimulation);
document.addEventListener('keydown', event=>{
  if(event.key === 'Escape' && document.getElementById('simulationOverlay').classList.contains('open')) closeSimulation();
});
// Metrics change only when a backend simulation result is applied.


/* === UNIVERSAL HOVER TOOLTIP SYSTEM - V4.2 PRESERVE ORIGINAL + REMOVE NATIVE TITLE === */
(function initRlFutureHoverV42(){
  const DETAILS = [
    ['.core-panel .panel-head span','Twin Hologram / 数字孪生全息态势','原信息：聚合 AGV、BESS 岸电储能、HVAC、堆场、靠泊窗口等状态，让管理者一眼判断港区稳不稳、风险集中在哪里。\n上海港口径：适配洋山四期自动化码头 + 上海港岸电保供场景。输入包括 ETA、泊位窗口、岸桥资源、AGV任务队列、BESS SoC、岸电负荷、堆场压力和能源价格。\n计算逻辑：所有指标先归一化到 0–100 或 0–1，再通过权重合成稳定度、风险压强和策略可信度。'],
    ['#systemMode','运行模式：SEMI-AUTO','原信息：半自动模式下，AI 负责生成策略、仿真、解释和风险提示，关键执行动作仍由人工确认。\n上海港解释：港口生产不能让 AI 黑箱自动下发关键控制，因此采用“AI 推荐 + dry-run + 人工审批 + 审计回放”的半自动闭环。'],
    ['.core-orb','中心孪生引擎','原信息：港口全局 AI 决策核心，汇总实时状态，预测未来风险，生成策略候选，并写入审计链。\n计算由来：中心分数来自五类输入：靠泊/泊位、AGV、能源/BESS、堆场、HVAC。每类先转成健康指数，再进入 RL 策略评估。'],
    ['.node-a','AGV 调度节点','原信息：监控 AGV 充换电、拥堵、任务队列与调度效率。\n上海港解释：对应洋山四期自动化运输链。\n计算口径：AGV压力 = 0.45×任务队列强度 + 0.35×换电/补能排队 + 0.20×路径冲突压力。当前显示 83，说明 AGV 是当前最应盯控节点。'],
    ['.node-b','BESS 岸电储能节点','原信息：监控 SoC 安全带、削峰窗口、岸电保供能力、电价与碳因子。\n上海港解释：岸电保供优先级高于削峰套利。\n计算口径：岸电/BESS压力 = 0.50×岸电负荷率 + 0.30×BESS保供压力 + 0.20×同时在泊受电需求。当前 71，表示可以优化但不能过度放电。'],
    ['.node-c','HVAC Cooling 节点','原信息：监控冷站负荷、温控约束、能耗曲线与设备安全边界。\n计算口径：HVAC可调空间 = 预冷余量、温控安全边界、压缩机效率和负荷峰谷差的综合结果。\n上海港解释：HVAC 是可调负荷，但只能温和错峰，不能牺牲设备和人员环境安全。'],
    ['.node-d','YARD 堆场节点','原信息：监控堆场压力、照明负荷、作业密度和异常区域。\n计算口径：堆场压力 = 0.45×箱位占用 + 0.35×闸口/集卡等待 + 0.20×翻箱强度。当前 64，说明可做分区照明和作业节奏优化，但不能影响安全照度。'],
    ['#stabilityValue','港区稳定度 · 92.4%','原信息：综合 AGV、岸电、堆场、冷站、靠泊窗口等状态形成的稳定评分。\n计算口径：92.4% = (靠泊保障93 + 能源保障94 + AGV可用性92 + 堆场连续性91 + 安全审计通过率92) / 5。\n上海港解释：这是一眼给管理层看的“全局健康分”，不是单个设备数值。当前属于高稳定区。'],
    ['#riskValue','风险压强 · LOW','原信息：表示未来 30–60 分钟可能出现的压力等级。\n计算口径：风险分 = 0.35×靠泊峰值风险0.36 + 0.30×BESS/岸电风险0.28 + 0.20×堆场拥堵风险0.31 + 0.15×模型漂移0.24 = 0.31。\n分级：<0.35 为 LOW，0.35–0.65 为 MEDIUM，>0.65 为 HIGH。'],
    ['#trustValue','策略可信度 · 0.87','原信息：由模型置信、约束通过率、反事实对比结果与历史回放一致性形成。\n计算口径：0.87 = 0.35×离线回放一致性0.89 + 0.25×护栏通过率1.00 + 0.20×反事实收益稳定性0.82 + 0.20×历史场景相似度0.73。\n解释：可信度高于 0.85，可进入 dry-run，但仍不建议跳过人工审批。'],
    ['.situation-panel .panel-head span','Situation Pulse / 实时态势脉冲','原信息：把靠泊窗口、堆场压力、岸电负荷、AGV 调度、碳强度压缩成五条态势条。\n上海港解释：这五项正好对应大型集装箱港的核心管理维度：泊位、堆场、能源、水平运输、绿色低碳。'],
    ['.insight-box','AI 总结','原信息：把复杂指标翻译成管理结论。\n计算由来：摘要优先读取最高压力项和最接近护栏的约束。当前最高项为 AGV 调度 83，其次为靠泊窗口 78 与岸电负荷 71，所以建议盯控靠泊高峰、BESS SoC 和堆场照明边界。'],
    ['.strategy-panel .panel-head span','RL 策略候选','原信息：强化学习不会只给单一答案，而是给保守、平衡、激进三类候选。\n计算逻辑：三类策略来自同一状态输入，但风险偏好不同：保守保稳定，平衡追求收益/风险折中，激进追求更高削峰收益但必须人工复核。'],
    ['.timeline','反事实时间线','原信息：每个亮点代表未来关键窗口，用于比较执行/不执行策略的差异。\n上海港解释：时间线对应靠泊、岸电、AGV换电、峰值负荷和审批回执等关键节点。'],
    ['.counter-grid .counter-item:nth-child(1)','若执行保守策略 · -6.8% 能耗','原信息：低风险，收益稳定。\n计算口径：基线 60 分钟综合电耗 12.50MWh，执行后 11.65MWh，节省 0.85MWh；0.85/12.50=6.8%。\n上海港解释：适合靠泊高峰前先稳住生产连续性。'],
    ['.counter-grid .counter-item:nth-child(2)','若执行平衡策略 · -10.5% 峰值','原信息：推荐进入 dry-run。\n计算口径：基线峰值 3.80MW，执行后预计 3.40MW；(3.80-3.40)/3.80=10.5%。\n上海港解释：适合在岸电保供充足、BESS SoC 安全的情况下执行。'],
    ['.counter-grid .counter-item:nth-child(3)','若执行激进策略 · +0.18 风险','原信息：需要人工复核。\n计算口径：风险分由 0.31 升至 0.49，因此净增 +0.18。\n上海港解释：它能进一步削峰，但会压缩 BESS SoC 和作业安全余量，只能短时审批执行。'],
    ['.guard-list','Guardrail Matrix / 执行护栏','原信息：所有策略必须先通过 SoC、安全、服务水平、异常漂移等护栏检查。\n计算逻辑：护栏不是软建议，而是执行前硬约束。任何一个硬约束不通过，策略不能进入执行。'],
    ['.guard-list li:nth-child(1)','SoC 安全带未越界','计算口径：BESS SoC 安全带设为 35%–85%，当前演示 SoC 约 61%，处于安全带中部。\n解释：平衡策略可执行，激进策略会更接近下边界。'],
    ['.guard-list li:nth-child(2)','岸电保障窗口可覆盖','计算口径：关键需求约 4.2MW，可用岸电 + 储能保障能力约 4.8MW，覆盖比 4.8/4.2=1.14。\n解释：覆盖比 >1 表示关键靠泊船舶受电需求可被保障。'],
    ['.guard-list li:nth-child(3)','服务水平不低于阈值','计算口径：仿真显示关键周转效率保持在基线 96% 以上，高于 95% 阈值。\n解释：节能和削峰不能以牺牲岸桥、AGV、堆场 SLA 为代价。'],
    ['.guard-list li:nth-child(4)','异常漂移已进入审计','计算口径：模型漂移指数约 0.24，低于 0.35 观察阈值。\n解释：当前仍属于相似历史场景，但异常输入会被记录，便于复盘。'],
    ['.agent-grid','MAS Swarm / 多智能体协同','原信息：Planner、Energy、Yard、Safety、Audit、Copilot 分别负责计划、能源、堆场、安全、审计和解释。\n上海港解释：大型港口决策不是单一模型能完成，需要多智能体按业务域分工协同。'],
    ['.loop-chain','HITL 执行闭环','原信息：推荐 → 仿真 → 审批 → 执行 → 回执 → 审计。\n计算/治理口径：策略只有在收益为正、风险未越界、护栏通过、人工审批完成后，才允许进入执行和回执记录。'],
    ['#consoleLog','执行日志','原信息：展示 AI 从读取状态、生成策略、通过护栏、反事实评估到等待人工确认的全过程。\n解释：这是演示“可追责”的关键，不是只给一个漂亮结果，而是展示每一步为什么能执行。'],
    ['.radar-shell','Carbon & Energy Radar / 碳能雷达','原信息：雷达扫描削峰、降碳、保供、成本四个方向。\n计算逻辑：综合评分 = 0.30×削峰 + 0.25×保供 + 0.25×成本 + 0.20×降碳。保供不达标时，一票否决。'],
    ['.radar-label.l1','削峰','原信息：关注峰值负荷、需量管理和储能放电窗口。\n计算口径：峰值从 3.80MW 降至 3.40MW，削峰率 10.5%。\n上海港解释：用于降低尖峰电费和配电压力。'],
    ['.radar-label.l2','降碳','原信息：结合碳因子、电价、岸电使用比例与设备调度，选择更低碳的运行时段和策略。\n计算口径：60 分钟窗口预计减排约 0.51 吨 CO2e，约 7.4%。\n解释：来自电耗下降和高碳时段用电后移。'],
    ['.radar-label.l3','保供','原信息：确保靠泊窗口和关键设备用电不被削弱。\n计算口径：可用保障能力 4.8MW / 关键岸电需求 4.2MW = 1.14。\n解释：保供是第一优先级，低于 1 不允许执行节能策略。'],
    ['.radar-label.l4','成本','原信息：综合电价、峰谷差、策略执行成本和潜在风险成本。\n计算口径：平衡策略预计 60 分钟电费下降约 8.1%。\n解释：来自削峰、错峰和能耗下降，但需扣除 BESS 损耗与寿命成本。']
  ];

  function style(el, obj){ Object.keys(obj).forEach(k=>el.style[k]=obj[k]); }
  function addTip(el,title,body){
    if(!el) return;
    el.dataset.tipTitle = title;
    el.dataset.tipBody = body;
    el.setAttribute('aria-label', title + '：' + body);
    el.removeAttribute('title');
    el.style.cursor = 'help';
  }
  function addHotspot(parentSelector, className, title, body, pos){
    const parent = document.querySelector(parentSelector);
    if(!parent) return;
    if(getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
    const h = document.createElement('div');
    h.className = 'rl-hotspot-v4 ' + className;
    h.dataset.tipTitle = title;
    h.dataset.tipBody = body;
    h.setAttribute('aria-label', title + '：' + body);
    style(h, Object.assign({
      position:'absolute', zIndex:'999', pointerEvents:'auto', cursor:'help',
      background:'rgba(94,231,255,0.001)', border:'1px solid transparent', borderRadius:'18px'
    }, pos));
    parent.appendChild(h);
  }
  function ensureTip(){
    let tip = document.getElementById('rlFutureTooltipV4');
    if(tip) return tip;
    tip = document.createElement('div');
    tip.id = 'rlFutureTooltipV4';
    tip.innerHTML = '<div class="rl-title"></div><div class="rl-body"></div>';
    style(tip, {
      position:'fixed', zIndex:'2147483647', left:'0px', top:'0px', width:'430px', maxWidth:'calc(100vw - 24px)',
      opacity:'0', transform:'translateY(8px) scale(.98)', transition:'opacity .12s ease, transform .12s ease',
      pointerEvents:'none', border:'1px solid rgba(103,255,178,.48)', borderRadius:'16px', padding:'13px 14px',
      background:'linear-gradient(180deg,rgba(5,18,36,.98),rgba(3,9,22,.98))', color:'#eaf7ff',
      boxShadow:'0 18px 44px rgba(0,0,0,.58),0 0 30px rgba(94,231,255,.22),inset 0 1px 0 rgba(255,255,255,.08)',
      backdropFilter:'blur(12px)'
    });
    style(tip.querySelector('.rl-title'), {fontSize:'13px', fontWeight:'900', color:'#dfffee', marginBottom:'7px', letterSpacing:'.04em'});
    style(tip.querySelector('.rl-body'), {fontSize:'12px', lineHeight:'1.7', color:'#bdefff', whiteSpace:'pre-line'});
    document.body.appendChild(tip);
    return tip;
  }
  function move(ev){
    const tip = ensureTip();
    const pad = 18;
    const rect = tip.getBoundingClientRect();
    const w = rect.width || 430;
    const h = rect.height || 180;
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if(x + w > innerWidth) x = ev.clientX - w - pad;
    if(y + h > innerHeight) y = ev.clientY - h - pad;
    tip.style.left = Math.max(10,x) + 'px';
    tip.style.top = Math.max(10,y) + 'px';
  }
  function show(el, ev){
    const tip = ensureTip();
    tip.querySelector('.rl-title').textContent = el.dataset.tipTitle || '';
    tip.querySelector('.rl-body').textContent = el.dataset.tipBody || '';
    tip.style.opacity = '1';
    tip.style.transform = 'translateY(0) scale(1)';
    move(ev);
  }
  function hide(){
    const tip = ensureTip();
    tip.style.opacity = '0';
    tip.style.transform = 'translateY(8px) scale(.98)';
  }
  function init(){
    DETAILS.forEach(([sel,t,b]) => document.querySelectorAll(sel).forEach(el => addTip(el,t,b)));
    addHotspot('.orbital-stage','hot-core','中心孪生引擎 · 计算汇总区','原信息：港口全局 AI 决策核心，汇总实时状态，生成未来风险判断，并驱动策略候选。\n计算口径：输入靠泊、AGV、BESS、HVAC、堆场五类归一化指标，再输出稳定度、风险压强和策略可信度。',{left:'37%',top:'32%',width:'26%',height:'34%'});
    addHotspot('.orbital-stage','hot-agv','AGV 联动风险','原信息：关注靠泊窗口变化对 AGV 调度、充换电队列和堆场压力的连锁影响。\n计算口径：83 = 0.45×任务队列强度86 + 0.35×换电排队81 + 0.20×路径冲突79。',{left:'4%',top:'9%',width:'32%',height:'31%'});
    addHotspot('.orbital-stage','hot-bess','岸电储能保供风险','原信息：关注 BESS SoC 是否足够覆盖靠泊高峰，以及是否存在峰值电价和负荷压力。\n计算口径：岸电覆盖比约 4.8MW / 4.2MW = 1.14；SoC 安全带 35%–85%，当前约 61%。',{right:'4%',top:'9%',width:'32%',height:'31%'});
    addHotspot('.orbital-stage','hot-hvac','HVAC 能耗优化窗口','原信息：判断冷站是否可在不越过安全边界的情况下降载、错峰或预冷。\n计算口径：看预冷余量、温控边界、压缩机效率和峰谷价差，当前只适合温和错峰。',{left:'4%',bottom:'7%',width:'32%',height:'31%'});
    addHotspot('.orbital-stage','hot-yard','堆场压力与照明策略','原信息：观察堆场作业密度、照明负荷和安全约束，决定是否执行节能策略。\n计算口径：64 = 0.45×箱位占用68 + 0.35×等待61 + 0.20×翻箱58。',{right:'4%',bottom:'7%',width:'32%',height:'31%'});
    addHotspot('.radar-shell','hot-radar-center','综合优化扫描区','原信息：雷达中心代表碳、能、成本、保供的综合平衡点。\n计算口径：综合评分 = 0.30×削峰 + 0.25×保供 + 0.25×成本 + 0.20×降碳；保供不达标一票否决。',{left:'31%',top:'24%',width:'38%',height:'50%'});
    addHotspot('.radar-shell','hot-radar-top','削峰压力区','原信息：上方区域对应峰值负荷和需量控制。\n计算口径：3.80MW → 3.40MW，削峰率 10.5%。',{left:'28%',top:'1%',width:'44%',height:'25%'});
    addHotspot('.radar-shell','hot-radar-right','降碳收益区','原信息：右侧区域对应碳因子变化和低碳调度收益。\n计算口径：60 分钟窗口减排约 0.51 吨 CO2e，约 7.4%。',{right:'1%',top:'27%',width:'28%',height:'46%'});
    addHotspot('.radar-shell','hot-radar-bottom','保供安全区','原信息：下方区域对应生产连续性和靠泊保障。\n计算口径：可用保障能力 4.8MW / 关键需求 4.2MW = 1.14，>1 才允许执行。',{left:'28%',bottom:'1%',width:'44%',height:'25%'});
    addHotspot('.radar-shell','hot-radar-left','成本控制区','原信息：左侧区域对应电价、峰谷差和执行成本。\n计算口径：平衡策略预计 60 分钟电费下降约 8.1%，但需扣除 BESS 损耗与寿命成本。',{left:'1%',top:'27%',width:'28%',height:'46%'});
    document.addEventListener('pointerover', function(ev){
      const el = ev.target.closest && ev.target.closest('[data-tip-title]');
      if(el) show(el, ev);
    }, true);
    document.addEventListener('pointermove', function(ev){
      const el = ev.target.closest && ev.target.closest('[data-tip-title]');
      if(el) move(ev);
    }, true);
    document.addEventListener('pointerout', function(ev){
      const from = ev.target.closest && ev.target.closest('[data-tip-title]');
      const to = ev.relatedTarget && ev.relatedTarget.closest && ev.relatedTarget.closest('[data-tip-title]');
      if(from && from !== to) hide();
    }, true);
    console.log('[rl_future] Universal hover tooltip system V4.4 title copy landing-ready loaded - native title removed. targets=', document.querySelectorAll('[data-tip-title]').length);
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
