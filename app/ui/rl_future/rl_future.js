const initialSituation = [
  ['BESS SoC', null],
  ['岸电功率', null],
  ['奖励漂移', null],
  ['候选数量', null],
  ['硬约束', null],
];

let strategies = [];
let futureRunActive = false;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function renderBars(items = initialSituation) {
  const root = document.getElementById('situationBars');
  root.innerHTML = items.map(([name, value, display]) => {
    const width = value == null ? 0 : Math.max(0, Math.min(100, finiteNumber(value)));
    return `
      <div class="bar-row">
        <span>${escapeHtml(name)}</span>
        <div class="bar-track" aria-label="${escapeHtml(name)}"><div class="bar-fill" style="width:${width}%"></div></div>
        <b>${escapeHtml(display ?? '--')}</b>
      </div>
    `;
  }).join('');
}

function renderStrategies() {
  const root = document.getElementById('strategyStack');
  if (!strategies.length) {
    root.innerHTML = [1, 2, 3].map(index => `
      <div class="strategy-card">
        <div class="strategy-title"><span>候选 ${index}</span><b>WAIT</b></div>
        <div class="strategy-meta">
          <span>节能 <b>--</b></span><span>风险 <b>--</b></span><span>可信度 <b>--</b></span>
        </div>
      </div>
    `).join('');
    return;
  }
  root.innerHTML = strategies.map(item => `
    <div class="strategy-card ${item.active ? 'active' : ''}">
      <div class="strategy-title"><span>${escapeHtml(item.name)}</span><b>${escapeHtml(item.tag)}</b></div>
      <div class="strategy-meta">
        <span>节能 <b>${escapeHtml(item.save)}</b></span>
        <span>风险 <b>${escapeHtml(item.risk)}</b></span>
        <span>可信度 <b>${escapeHtml(item.trust)}</b></span>
      </div>
    </div>
  `).join('');
}

function renderTimeline() {
  const root = document.getElementById('timeline');
  root.innerHTML = [12, 32, 53, 76, 91]
    .map((left, index) => `<span class="tick" style="left:${left}%;animation-delay:${index * 0.22}s" aria-label="推演检查点 ${index + 1}"></span>`)
    .join('');
}

function writeLogs(sourceLogs = []) {
  const box = document.getElementById('consoleLog');
  box.style.whiteSpace = 'pre-wrap';
  box.textContent = sourceLogs.length
    ? sourceLogs.join('\n')
    : [
        '[Boundary] 当前为离线工程推演页，不代表港口实时态势',
        '[Evidence] 未收到后端结果前不展示收益、风险或可信度数值',
        '[Safety] 推演接口不连接生产设备，执行下发默认关闭',
      ].join('\n');
}

function setStage(stageId, state) {
  const element = document.querySelector(`.run-stage[data-stage="${stageId}"]`);
  if (!element) return;
  element.classList.remove('pending', 'active', 'done', 'blocked');
  element.classList.add(state);
  const label = element.querySelector('em');
  if (label) {
    label.textContent = ({pending: 'WAIT', active: 'RUN', done: 'PASS', blocked: 'BLOCK'})[state] || state;
  }
}

function appendTerminal(message, tone = '') {
  const terminal = document.getElementById('simulationTerminal');
  const idle = terminal.querySelector('.terminal-idle');
  if (idle) idle.remove();
  const line = document.createElement('div');
  line.className = `terminal-line ${tone}`.trim();
  const text = document.createElement('span');
  text.textContent = message;
  line.appendChild(text);
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

function resetRunUI() {
  ['situation', 'candidates', 'counterfactual', 'guardrails', 'receipt']
    .forEach(id => setStage(id, 'pending'));
  document.getElementById('runIdValue').textContent = '正在请求';
  document.getElementById('snapshotStatus').textContent = '待读取';
  document.getElementById('candidateStatus').textContent = '0 / 0';
  document.getElementById('overlayGuardStatus').textContent = '等待校验';
  document.getElementById('simulationTerminal').innerHTML =
    '<div class="terminal-idle">等待 FastAPI 返回本次推演证据……</div>';
  document.getElementById('snapshotGrid').innerHTML = [
    ['BESS 荷电状态', '等待后端场景快照'],
    ['岸电当前功率', '等待后端场景快照'],
    ['模型奖励漂移', '等待可观测信号'],
    ['策略候选池', '等待候选生成'],
  ].map(([name, note]) => `<div><span>${name}</span><strong>--</strong><em>${note}</em></div>`).join('');
  document.getElementById('runCandidateGrid').innerHTML =
    '<div class="candidate-placeholder"></div><div class="candidate-placeholder"></div><div class="candidate-placeholder"></div>';
  document.getElementById('runGuardGrid').innerHTML =
    '<div class="guard-placeholder"></div><div class="guard-placeholder"></div><div class="guard-placeholder"></div>';
  const decision = document.getElementById('decisionModule');
  decision.classList.remove('ready', 'blocked');
  document.getElementById('decisionLabel').textContent = '等待后端推演';
  document.getElementById('decisionReason').textContent =
    '结果返回前不生成推荐结论，也不会进入任何执行步骤。';
  document.getElementById('recommendedStrategy').textContent = '等待计算';
  document.getElementById('evidenceDigest').textContent = '等待生成';
  document.getElementById('productionBoundary').textContent =
    '只做候选、仿真、护栏与审计，不向生产设备下发控制指令。';
  document.getElementById('finishSimulation').disabled = true;
}

function openSimulation() {
  resetRunUI();
  const overlay = document.getElementById('simulationOverlay');
  overlay.classList.add('open');
  overlay.setAttribute('aria-hidden', 'false');
  document.body.classList.add('simulation-active');
}

function closeSimulation() {
  const overlay = document.getElementById('simulationOverlay');
  overlay.classList.remove('open');
  overlay.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('simulation-active');
  document.getElementById('btnIgnite').focus();
}

function renderSnapshot(snapshot) {
  const items = [
    ['BESS 荷电状态', `${finiteNumber(snapshot.bess_soc_pct).toFixed(1)}%`, '后端场景快照'],
    ['岸电当前功率', `${finiteNumber(snapshot.shore_power_kw).toFixed(0)} kW`, '岸电节点聚合'],
    ['模型奖励漂移', finiteNumber(snapshot.reward_drift).toFixed(3), 'RL Ops 观测信号'],
    ['策略候选池', `${finiteNumber(snapshot.candidate_pool_size).toFixed(0)} 条`, '候选生成器输出'],
  ];
  document.getElementById('snapshotGrid').innerHTML = items.map((item, index) => `
    <div class="revealed" style="animation-delay:${index * 0.08}s">
      <span>${item[0]}</span><strong>${item[1]}</strong><em>${item[2]}</em>
    </div>
  `).join('');
  document.getElementById('snapshotStatus').textContent =
    `${finiteNumber(snapshot.horizon_min, 90)} MIN · OFFLINE`;
}

function renderRunCandidates(data) {
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const root = document.getElementById('runCandidateGrid');
  root.innerHTML = candidates.map((item, index) => {
    const baseline = finiteNumber(item.baseline_energy_kwh);
    const energySaving = finiteNumber(item.energy_saving_kwh);
    const energyPercent = baseline > 0 ? energySaving / baseline * 100 : 0;
    const recommended = item.id === data.recommended_strategy_id;
    return `<article class="run-candidate ${recommended ? 'recommended' : ''} ${item.dispatch_ready ? '' : 'blocked'}" style="animation-delay:${index * 0.12}s">
      <div class="candidate-top">
        <div><span class="candidate-mode">${escapeHtml(item.mode || 'SCENARIO')}</span><h4>${escapeHtml(item.title || item.id)}</h4></div>
        <b class="candidate-tag">${recommended ? '推荐' : escapeHtml(item.tag || 'CANDIDATE')}</b>
      </div>
      <div class="candidate-metrics">
        <div><span>节能</span><b>${energySaving.toFixed(1)} kWh</b></div>
        <div><span>削峰</span><b>${finiteNumber(item.peak_reduction_kw).toFixed(1)} kW</b></div>
        <div><span>可信度</span><b>${finiteNumber(item.confidence).toFixed(2)}</b></div>
      </div>
      <div class="candidate-result"><span>电耗改善 ${energyPercent.toFixed(1)}%</span><strong>${item.dispatch_ready ? '仿真可用' : '保持阻断'}</strong></div>
    </article>`;
  }).join('') || '<div class="terminal-idle">后端未返回候选策略。</div>';
  document.getElementById('candidateStatus').textContent = `${candidates.length} / ${candidates.length}`;
}

function renderRunGuardrails(data) {
  const guardrails = Array.isArray(data.guardrails) ? data.guardrails : [];
  document.getElementById('runGuardGrid').innerHTML = guardrails.map((item, index) => `
    <article class="run-guard ${item.passed ? 'pass' : 'block'}" style="animation-delay:${index * 0.08}s">
      <div class="guard-state"><span>${item.level === 'hard' ? 'HARD' : 'SOFT'}</span><b>${item.passed ? 'PASS' : 'BLOCK'}</b></div>
      <h4>${escapeHtml(item.name)}</h4>
      <div class="guard-measure">${escapeHtml(item.actual)}${escapeHtml(item.unit || '')}</div>
      <span class="guard-threshold">阈值 ${escapeHtml(item.threshold)}</span>
      <span class="guard-source">${escapeHtml(item.source)}</span>
    </article>
  `).join('');
  const hardRules = guardrails.filter(item => item.level === 'hard');
  const hardPassed = hardRules.length > 0 && hardRules.every(item => item.passed);
  document.getElementById('overlayGuardStatus').textContent =
    hardPassed ? 'HARD RULES PASS' : 'HARD RULE BLOCKED';
}

function renderDecision(data) {
  const ready = Boolean(data.decision && data.decision.ready_for_human_dry_run);
  const module = document.getElementById('decisionModule');
  module.classList.remove('ready', 'blocked');
  module.classList.add(ready ? 'ready' : 'blocked');
  document.getElementById('decisionLabel').textContent = data.decision.label;
  document.getElementById('decisionReason').textContent =
    `${data.decision.next_action}。${data.decision.production_boundary}`;
  document.getElementById('recommendedStrategy').textContent =
    data.decision.recommended_strategy_title || '无可用策略';
  document.getElementById('evidenceDigest').textContent = data.audit.evidence_digest;
  document.getElementById('productionBoundary').textContent =
    data.decision.production_boundary;
}

function applyRunToMainSurface(data) {
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const guardrails = Array.isArray(data.guardrails) ? data.guardrails : [];
  const recommended =
    candidates.find(item => item.id === data.recommended_strategy_id) || candidates[0];
  const hardRules = guardrails.filter(item => item.level === 'hard');
  const hardPassCount = hardRules.filter(item => item.passed).length;
  const hardPassRate = hardRules.length ? hardPassCount / hardRules.length * 100 : 0;

  document.getElementById('stabilityValue').textContent =
    hardRules.length ? `${hardPassRate.toFixed(0)}%` : '--';
  document.getElementById('riskValue').textContent =
    data.decision.ready_for_human_dry_run ? 'DRY-RUN' : 'BLOCKED';
  document.getElementById('trustValue').textContent =
    recommended ? finiteNumber(recommended.confidence).toFixed(2) : '--';

  const driftLimitGuard = guardrails.find(item => item.id === 'model_drift');
  renderBars([
    ['BESS SoC', finiteNumber(data.snapshot.bess_soc_pct), `${finiteNumber(data.snapshot.bess_soc_pct).toFixed(0)}%`],
    ['岸电功率', Math.min(100, finiteNumber(data.snapshot.shore_power_kw) / 50), `${finiteNumber(data.snapshot.shore_power_kw).toFixed(0)}kW`],
    ['奖励漂移', Math.min(100, finiteNumber(data.snapshot.reward_drift) * 1000), finiteNumber(data.snapshot.reward_drift).toFixed(3)],
    ['候选数量', Math.min(100, candidates.length / 3 * 100), String(candidates.length)],
    ['硬约束', hardPassRate, `${hardPassCount}/${hardRules.length}`],
  ]);

  strategies = candidates.map(item => {
    const baseline = finiteNumber(item.baseline_energy_kwh);
    const savingPercent = baseline > 0 ? finiteNumber(item.energy_saving_kwh) / baseline * 100 : 0;
    return {
      name: item.title || item.id,
      tag: item.id === data.recommended_strategy_id ? 'RECOMMENDED' : (item.tag || 'CANDIDATE'),
      save: `${savingPercent.toFixed(1)}%`,
      risk: item.risk_level || 'UNKNOWN',
      trust: finiteNumber(item.confidence).toFixed(2),
      active: item.id === data.recommended_strategy_id,
    };
  });
  renderStrategies();

  document.getElementById('counterGrid').innerHTML = candidates.map(item => `
    <div class="counter-item">
      <span>${escapeHtml(item.mode || '候选')} · ${escapeHtml(item.title || item.id)}</span>
      <strong>节能 ${finiteNumber(item.energy_saving_kwh).toFixed(1)} kWh</strong>
      <em>削峰 ${finiteNumber(item.peak_reduction_kw).toFixed(1)} kW · ${item.dispatch_ready ? '仿真可用' : '护栏前阻断'}</em>
    </div>
  `).join('');
  document.getElementById('guardList').innerHTML = guardrails.map(item => `
    <li class="${item.passed ? 'pass' : 'block'}"><span></span>${escapeHtml(item.name)} · ${item.passed ? '通过' : '阻断'}</li>
  `).join('');
  const guardStatus = document.getElementById('guardStatus');
  guardStatus.textContent = data.decision.ready_for_human_dry_run ? 'DRY-RUN' : 'BLOCKED';
  guardStatus.style.background =
    data.decision.ready_for_human_dry_run ? 'var(--green)' : '#ff6c7f';
  document.getElementById('aiSummary').textContent =
    `${data.decision.label}。推荐候选：${data.decision.recommended_strategy_title || '无'}。${data.decision.production_boundary}` +
    (driftLimitGuard ? ` 奖励漂移检查：${driftLimitGuard.passed ? '通过' : '阻断'}。` : '');
  writeLogs(data.logs || []);
}

async function ignite() {
  if (futureRunActive) return;
  futureRunActive = true;
  const button = document.getElementById('btnIgnite');
  button.disabled = true;
  button.textContent = '请求后端…';
  openSimulation();
  setStage('situation', 'active');
  appendTerminal('[Boundary] 建立离线反事实推演通道；生产下发接口保持隔离', 'audit');
  try {
    const response = await fetch('/api/rl/future/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        horizon_min: 90,
        step_min: 5,
        max_candidates: 3,
        source: 'rl-future-deck',
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `推演接口返回 ${response.status}`);
    }
    document.getElementById('runIdValue').textContent = data.run_id;
    renderSnapshot(data.snapshot);
    renderRunCandidates(data);
    renderRunGuardrails(data);
    renderDecision(data);
    (data.logs || []).forEach(line => appendTerminal(
      line,
      line.includes('BLOCK') || line.includes('阻断') ? 'block' :
        line.includes('[Audit]') || line.includes('[Boundary]') ? 'audit' : 'pass',
    ));
    (data.stages || []).forEach(stage => {
      setStage(stage.id, stage.status === 'blocked' ? 'blocked' : 'done');
    });
    applyRunToMainSurface(data);
    document.getElementById('finishSimulation').disabled = false;
    button.textContent = '再次推演';
  } catch (error) {
    const activeStage = document.querySelector('.run-stage.active');
    if (activeStage) setStage(activeStage.dataset.stage, 'blocked');
    appendTerminal(`[Fail Closed] ${error.message}`, 'block');
    appendTerminal('[Boundary] 推演未完成，未产生任何生产下发动作', 'audit');
    const module = document.getElementById('decisionModule');
    module.classList.add('blocked');
    document.getElementById('decisionLabel').textContent = '推演失败，保持安全阻断';
    document.getElementById('decisionReason').textContent =
      `${error.message}。系统未调用生产下发接口，请检查服务后重新推演。`;
    document.getElementById('runIdValue').textContent = 'FAILED-CLOSED';
    document.getElementById('finishSimulation').disabled = false;
    button.textContent = '重新推演';
  } finally {
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
document.addEventListener('keydown', event => {
  if (
    event.key === 'Escape' &&
    document.getElementById('simulationOverlay').classList.contains('open')
  ) {
    closeSimulation();
  }
});
