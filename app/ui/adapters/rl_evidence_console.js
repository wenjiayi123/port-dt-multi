(function(){
  if(window.__rlEvidenceConsoleInstalled || window.location.pathname !== "/rl-panel") return;
  window.__rlEvidenceConsoleInstalled = true;

  const FACTOR_LABELS = {
    wind_speed_mps:"风速", visibility_km:"能见度", wave_height_m:"浪高",
    current_speed_mps:"流速", berth_occupancy_ratio:"泊位占用",
    yard_occupancy_ratio:"堆场占用", crane_availability_ratio:"岸桥可用",
    equipment_availability_ratio:"设备可用", channel_congestion_ratio:"航道拥堵",
    reefer_load_kw:"冷藏箱负荷", pilot_tug_availability_ratio:"引航/拖轮可用",
    closure_flag:"港区关闭"
  };

  function escapeHtml(value){
    return String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[char]));
  }

  function addStyle(){
    const style = document.createElement("style");
    style.textContent = `
      .rl-proof-console{margin:16px 18px 0;border:1px solid rgba(56,189,248,.46);border-radius:18px;background:radial-gradient(circle at 8% 0%,rgba(14,165,233,.18),transparent 28%),linear-gradient(145deg,#081526,#07101f 62%,#0b1628);box-shadow:0 24px 65px rgba(2,8,23,.38);padding:16px;color:#e7f3ff}
      .rl-proof-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}
      .rl-proof-title{font-size:21px;font-weight:950;letter-spacing:.2px}.rl-proof-title small{display:block;margin-top:5px;color:#7dd3fc;font-size:11px;letter-spacing:1.4px;text-transform:uppercase}
      .rl-proof-actions{display:flex;gap:8px;flex-wrap:wrap}.rl-proof-btn{border:1px solid #31527a;border-radius:10px;background:#102540;color:#d9efff;padding:8px 11px;font-weight:850;cursor:pointer}.rl-proof-btn.primary{background:linear-gradient(180deg,#0ea5e9,#0369a1);border-color:#38bdf8;color:white}
      .rl-proof-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:14px}.rl-proof-kpi{border:1px solid #223a59;border-radius:12px;background:rgba(5,18,35,.82);padding:10px}.rl-proof-kpi span{display:block;color:#8db6d8;font-size:10px}.rl-proof-kpi b{display:block;margin-top:5px;color:#f1f8ff;font-size:18px}.rl-proof-kpi em{display:block;margin-top:4px;color:#7dd3fc;font-size:10px;font-style:normal}
      .rl-proof-grid{display:grid;grid-template-columns:1.15fr 1fr .9fr;gap:10px;margin-top:10px}.rl-proof-panel{border:1px solid #223a59;border-radius:13px;background:rgba(4,14,29,.84);padding:12px;min-width:0}.rl-proof-panel h3{margin:0 0 8px;color:#eaf7ff;font-size:13px}.rl-proof-panel h3 small{color:#7f9fbd;font-size:10px;margin-left:5px}
      .rl-algo-table,.rl-data-table{width:100%;border-collapse:collapse;font-size:10px}.rl-algo-table th,.rl-algo-table td,.rl-data-table th,.rl-data-table td{border-bottom:1px solid #1a304b;padding:6px 5px;text-align:left;vertical-align:top}.rl-algo-table th,.rl-data-table th{color:#7fa7c9;font-weight:800}.rl-algo-table td,.rl-data-table td{color:#cfe7fa}.rl-proof-status{display:inline-flex;border-radius:999px;padding:2px 6px;border:1px solid #3b5471;color:#a8c5dc}.rl-proof-status.ok{color:#86efac;border-color:#217a4c;background:rgba(22,101,52,.17)}.rl-proof-status.warn{color:#fde68a;border-color:#8a5b17;background:rgba(113,63,18,.18)}
      .rl-contract-row{display:grid;grid-template-columns:76px 1fr;gap:8px;padding:6px 0;border-bottom:1px solid #1a304b;font-size:10px}.rl-contract-row span{color:#7fa7c9}.rl-contract-row b{color:#d7ebfb;line-height:1.5}.rl-factor-cloud{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.rl-factor-cloud i{font-style:normal;font-size:9px;color:#9cd8f5;border:1px solid #244966;background:#071c31;border-radius:999px;padding:3px 6px}
      .rl-xiaoyi-card{display:grid;grid-template-columns:72px 1fr;gap:10px;align-items:center}.rl-xiaoyi-card img{width:72px;height:104px;object-fit:contain;filter:drop-shadow(0 9px 14px rgba(14,165,233,.22))}.rl-xiaoyi-card strong{display:block;color:#f2f9ff;font-size:13px}.rl-xiaoyi-card p{margin:5px 0 0;color:#99b7d0;font-size:10px;line-height:1.5}.rl-xiaoyi-advice{margin-top:8px;padding:8px;border:1px solid #28506d;border-radius:9px;background:#071b2e;color:#bae6fd;font-size:10px;line-height:1.55;min-height:48px}
      .rl-data-wrap{grid-column:1/-1}.rl-data-table td:nth-child(2),.rl-data-table td:nth-child(3){font-variant-numeric:tabular-nums}.rl-data-note{margin-top:7px;color:#83a7c4;font-size:9px;line-height:1.5}
      @media(max-width:1100px){.rl-proof-grid{grid-template-columns:1fr 1fr}.rl-data-wrap{grid-column:1/-1}.rl-proof-panel.xiaoyi{grid-column:1/-1}.rl-proof-kpis{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.rl-proof-head{display:block}.rl-proof-actions{margin-top:10px}.rl-proof-grid{grid-template-columns:1fr}.rl-proof-panel,.rl-data-wrap,.rl-proof-panel.xiaoyi{grid-column:1}.rl-proof-kpis{grid-template-columns:1fr 1fr}}
    `;
    document.head.appendChild(style);
  }

  function build(){
    const root = document.createElement("section");
    root.className = "rl-proof-console";
    root.id = "rlProofConsole";
    root.innerHTML = `
      <div class="rl-proof-head">
        <div class="rl-proof-title">训练中心 · 落地证据总览<small>TRAINING CENTER · ALGORITHM MATRIX · XIAOYI ADVISOR</small></div>
        <div class="rl-proof-actions">
          <button class="rl-proof-btn" data-proof-refresh>刷新后端证据</button>
          <button class="rl-proof-btn" data-proof-xiaoyi>打开小懿全系统助手</button>
          <button class="rl-proof-btn primary" data-proof-advice>小懿解读训练门禁</button>
        </div>
      </div>
      <div class="rl-proof-kpis">
        <div class="rl-proof-kpi"><span>真实控制器矩阵</span><b data-proof-algo>—</b><em>6 RL + 1 MPC</em></div>
        <div class="rl-proof-kpi"><span>当前公开训练包</span><b data-proof-rows>—</b><em data-proof-tier>等待读取来源</em></div>
        <div class="rl-proof-kpi"><span>v2 观测 / 动作</span><b data-proof-contract>—</b><em>缺失因素使用显式可用性掩码</em></div>
        <div class="rl-proof-kpi"><span>可用于比较结论的算法</span><b data-proof-ready>—</b><em>RL 需 ≥ 3 独立种子且每次 ≥ 10k 步</em></div>
      </div>
      <div class="rl-proof-grid">
        <div class="rl-proof-panel">
          <h3>算法矩阵 <small>真实实现与证据门禁</small></h3>
          <table class="rl-algo-table"><thead><tr><th>算法</th><th>实现</th><th>动作</th><th>种子/状态</th></tr></thead><tbody data-proof-algos><tr><td colspan="4">读取中…</td></tr></tbody></table>
        </div>
        <div class="rl-proof-panel">
          <h3>目标函数 · 观测 · 动作 <small>port_ops_v2</small></h3>
          <div class="rl-contract-row"><span>目标函数</span><b data-proof-objectives>从当前数据集的港口场景包读取中…</b></div>
          <div class="rl-contract-row"><span>基础观测</span><b>时钟、负荷、吞吐、到港、潮位、价格、碳因子、气温、SOC、队列、上一动作、时段进度</b></div>
          <div class="rl-contract-row"><span>建议动作</span><b>BESS功率、服务强度、柔性负荷、泊位优先级、堆场流量；默认无生产下发权</b></div>
          <div class="rl-factor-cloud" data-proof-factors></div>
        </div>
        <div class="rl-proof-panel xiaoyi">
          <h3>小懿训练顾问 <small>按钮—接口—门禁联动</small></h3>
          <div class="rl-xiaoyi-card">
            <img src="/static/xiaoyi_maritime_officer.png?v=20260725-q" alt="小懿Q版海事训练顾问">
            <div><strong>小懿AI · 训练与全系统助手</strong><p>读取后端数据指纹、算法实现、多种子门禁和场景包状态；训练按钮仍保留人工确认。</p></div>
          </div>
          <div class="rl-xiaoyi-advice" data-proof-advice-text>等待后端证据。点击“解读训练门禁”后，小懿只解释真实状态，不生成优秀指标。</div>
        </div>
        <div class="rl-proof-panel rl-data-wrap">
          <h3>公开数据包可信度对照 <small>行数不等于独立信息量</small></h3>
          <table class="rl-data-table"><thead><tr><th>数据集</th><th>训练行</th><th>独立官方观测</th><th>证据等级</th><th>因素覆盖</th><th>场景包 / SHA-256</th></tr></thead><tbody data-proof-datasets><tr><td colspan="6">读取中…</td></tr></tbody></table>
          <div class="rl-data-note">原新加坡数据、模型登记和业务KPI证据全部保留。新增洛杉矶包用于高频公开观测对照；二者都不是码头生产遥测，只有通过现场字段映射、来源授权、校准、影子运行和安全验收后才能形成现场结论。</div>
        </div>
      </div>
    `;
    const banner = document.getElementById("returnBanner");
    if(banner) banner.before(root); else document.body.querySelector("header")?.after(root);
    return root;
  }

  async function readJson(url){
    const response = await fetch(url, {cache:"no-store"});
    if(!response.ok) throw new Error(`${url} · HTTP ${response.status}`);
    return response.json();
  }

  function formatInteger(value){
    return Number(value || 0).toLocaleString("en-US");
  }

  function render(root, capabilities, datasetsPayload, benchmark){
    const algorithms = capabilities.algorithms || [];
    const summaries = new Map((benchmark.algorithms || []).map(item => [item.id, item]));
    root.querySelector("[data-proof-algo]").textContent = `${algorithms.length} 类`;
    const contract = capabilities.contracts?.port_ops_v2 || {};
    root.querySelector("[data-proof-contract]").textContent = `${contract.observation_dimensions || "—"}D / ${contract.continuous_action_dimensions || "—"}D`;
    const readyCount = [...summaries.values()].filter(item => item.multi_seed_ready).length;
    root.querySelector("[data-proof-ready]").textContent = `${readyCount} / ${algorithms.length}`;
    root.querySelector("[data-proof-algos]").innerHTML = algorithms.map(item => {
      const summary = summaries.get(item.id) || {};
      const seeds = summary.distinct_seeds || [];
      const ready = !!summary.multi_seed_ready;
      const status = ready ? "结论门禁通过" : (summary.smoke_runs ? "仅链路烟测" : "待正式评测");
      return `<tr><td><b>${escapeHtml(item.name)}</b><br><span class="rl-proof-status ${ready ? "ok" : "warn"}">${escapeHtml(item.family)}</span></td><td>${escapeHtml(item.implementation)}</td><td>${escapeHtml(item.action_space)}</td><td>${escapeHtml(seeds.join(", ") || "—")}<br><span class="rl-proof-status ${ready ? "ok" : "warn"}">${status}</span></td></tr>`;
    }).join("");
    root.querySelector("[data-proof-factors]").innerHTML = Object.keys(FACTOR_LABELS).map(name => `<i>${FACTOR_LABELS[name]}</i>`).join("");
    const datasets = datasetsPayload.datasets || [];
    const preferred = datasets.find(item => item.dataset_id === "public_us_la_6min_v1") || datasets.find(item => item.dataset_id === "public_port_ops_v1") || datasets[0] || {};
    const profiles = capabilities.port_profiles || [];
    const profile = profiles.find(item => item.profile_id === preferred.port_profile_id) || {};
    const objectives = profile.objectives || {};
    const objectiveLabels = {cost:"成本", carbon:"碳", peak:"峰值", safety:"安全", delay:"延迟"};
    const objectiveText = Object.entries(objectiveLabels)
      .map(([name, label]) => `${label} ${Math.round(Number(objectives[name] || 0) * 100)}%`)
      .join(" · ");
    root.querySelector("[data-proof-objectives]").textContent = objectiveText
      ? `${profile.profile_id || "reference_port_v1"}：${objectiveText}`
      : "场景包目标权重未登记";
    root.querySelector("[data-proof-rows]").textContent = formatInteger(preferred.rows);
    root.querySelector("[data-proof-tier]").textContent = preferred.quality?.evidence?.tier || "来源等级未登记";
    root.querySelector("[data-proof-datasets]").innerHTML = datasets.map(item => {
      const evidence = item.quality?.evidence || {};
      const factorCoverage = item.quality?.factor_coverage || {};
      const covered = Object.values(factorCoverage).filter(value => Number(value) > 0).length;
      const independent = Number(evidence.independent_source_observations || 0);
      const hash = String(item.sha256 || item.quality?.dataset_sha256 || "—");
      return `<tr data-dataset="${escapeHtml(item.dataset_id)}"><td><b>${escapeHtml(item.dataset_id)}</b><br>${escapeHtml(item.title || "")}</td><td>${formatInteger(item.rows)}</td><td>${independent ? formatInteger(independent) : "未登记"}</td><td>${escapeHtml(evidence.tier || "未登记")}</td><td>${covered} / ${Object.keys(FACTOR_LABELS).length}</td><td>${escapeHtml(item.port_profile_id || "reference_port_v1")}<br><span class="mono">${escapeHtml(hash.slice(0,12))}…</span></td></tr>`;
    }).join("") || '<tr><td colspan="6">没有可训练数据集</td></tr>';
  }

  function advise(root, capabilities, benchmark){
    const datasets = capabilities.datasets || [];
    const highFrequency = datasets.find(item => item.dataset_id === "public_us_la_6min_v1");
    const missing = (benchmark.algorithms || []).filter(item => !item.multi_seed_ready).map(item => item.name);
    const message = [
      highFrequency
        ? `已识别高频公开包 ${highFrequency.dataset_id}（${formatInteger(highFrequency.rows)} 行）。`
        : "高频洛杉矶公开包尚未构建，当前只可使用既有新加坡对照包。",
      missing.length
        ? `仍未通过多种子结论门禁：${missing.join("、")}。训练中心可运行，但不能把烟测写成性能结论。`
        : "七类控制器的持久化评测门禁已满足；仍需检查同数据指纹、同窗口与业务约束后再做比较。",
      "生产执行保持关闭；真实港口接入还需场景包校准、现场字段来源、影子运行和独立硬件联锁。"
    ].join("\n");
    root.querySelector("[data-proof-advice-text]").textContent = message;
  }

  async function load(root){
    root.querySelector("[data-proof-advice-text]").textContent = "正在读取后端能力、数据质量和持久化评测登记…";
    try{
      const [capabilities, datasets, benchmark] = await Promise.all([
        readJson("/api/rl/engine/capabilities"),
        readJson("/api/rl/datasets"),
        readJson("/api/rl/benchmarks/summary?dataset_id=public_us_la_6min_v1")
      ]);
      root.__proofState = {capabilities, datasets, benchmark};
      render(root, capabilities, datasets, benchmark);
      root.querySelector("[data-proof-advice-text]").textContent = "证据已同步。点击“解读训练门禁”查看当前可声明与不可声明的边界。";
    }catch(error){
      root.querySelector("[data-proof-advice-text]").textContent = `后端证据读取失败：${error.message}`;
    }
  }

  function install(){
    addStyle();
    const root = build();
    root.querySelector("[data-proof-refresh]").addEventListener("click", () => load(root));
    root.querySelector("[data-proof-advice]").addEventListener("click", () => {
      const state = root.__proofState;
      if(state) advise(root, state.capabilities, state.benchmark);
    });
    root.querySelector("[data-proof-xiaoyi]").addEventListener("click", () => {
      const widget = document.querySelector(".xiaoyi-sprite-root");
      if(widget){
        widget.classList.add("open");
        widget.querySelector(".xiaoyi-sprite-input")?.focus();
      }
    });
    load(root);
  }

  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", install);
  else install();
})();
