(function(){
  if (window.__portBilingualUiInstalled) return;
  window.__portBilingualUiInstalled = true;

  const VERSION = "2026-07-15-zh-headings-v5";
  const TITLE_LIKE_SELECTOR = [
    "h1", "h2", "h3", "h4", "h5", "h6", ".title", ".subtitle",
    ".home-lead", ".triad-label", ".dash-title", ".panel-title",
    ".mini-title", ".app-title", ".pm-layer-title", ".contract-name",
    ".contract-detail-title"
  ].join(",");
  const exact = new Map([
    ["首页总入口 · 先结论后细节", "Home Entry · Decisions First"],
    ["当前港区整体状态稳定，风险可控，优先处理高峰窗口与执行闭环回执。", "Port operations are stable and risks are controlled. Prioritize peak windows and execution-loop receipts."],
    ["这一屏先只回答四件事：当前运行是否稳、风险是否正在抬升、最值得优先处理的动作是什么、下一步该进入哪条主链路。更细的孪生、策略、执行与审计继续留在下方模块，首屏只负责做平台级判断与引导。", "This overview answers four questions first: operating health, rising risk, the highest-priority action, and the next workflow. Detailed twin, strategy, execution, and audit evidence continues below."],
    ["整体状态", "Overall Status"], ["风险热度", "Risk Level"], ["优先动作", "Priority Action"],
    ["稳定", "Stable"], ["需盯", "Watch"], ["需关注", "Needs Attention"], ["中低", "Low–Medium"], ["中高", "Medium–High"], ["低", "Low"], ["中", "Medium"], ["高", "High"],
    ["先看执行闭环", "Review Execution Loop First"], ["执行闭环", "Execution Loop"],
    ["数据源就绪态 / Contract Registry", "Data Readiness · Contract Registry"],
    ["管理驾驶舱（董事会 / CEO 视角）", "Executive Cockpit · Board / CEO View"],
    ["聚焦：钱 · 碳 · 风险 · 自动化程度", "Focus: Cost · Carbon · Risk · Automation"],
    ["年度节省电费（预估）", "Estimated Annual Electricity Savings"],
    ["年度减排 CO₂", "Annual CO₂ Reduction"],
    ["峰值需量风险（未来 30 天）", "Peak-Demand Risk · Next 30 Days"],
    ["自动闭环覆盖", "Automated Closed-Loop Coverage"],
    ["AI 策略可信度等级", "AI Strategy Confidence"],
    ["平台总体状态", "Platform Health"],
    ["平台总览地图", "Platform Overview Map"],
    ["开发生态 / App Center", "Developer Ecosystem · App Center"],
    ["多港区复用总览", "Multi-Port Reuse Overview"],
    ["ESG 总览", "ESG Overview"],
    ["合规报表 / 审计口径", "Compliance Reports · Audit Basis"],
    ["实时视图", "Real-Time View"], ["核心孪生", "Core Digital Twin"],
    ["策略编排总览", "Strategy Orchestration"], ["强化学习运营", "RL Operations"],
    ["多智能体协同", "Multi-Agent Coordination"], ["训练中心", "Training Center"],
    ["港口业务深耕", "Port Operations Intelligence"], ["堆场照明", "Yard Lighting"],
    ["暖通空调", "HVAC Cooling"], ["岸电储能", "Shore Power & BESS"],
    ["储能能量调度", "BESS Energy Dispatch"], ["场桥作业", "Yard Crane Operations"],
    ["AI 可信度总览", "AI Trust Overview"], ["异常检测 / 漂移", "Anomaly & Drift Monitoring"],
    ["OpsX 运行治理", "OpsX Runtime Governance"], ["外部信号", "External Signals"],
    ["MLOps 模型治理", "MLOps Model Governance"], ["治理审计", "Governance Audit"],
    ["管理驾驶舱", "Executive Cockpit"], ["多港区", "Multi-Port"], ["数字孪生", "Digital Twin"],
    ["策略 → 执行", "Strategy to Execution"], ["审计 / OpsX", "Audit & OpsX"],
    ["运营副驾", "Ops Copilot"], ["强化学习面板", "RL Panel"], ["项目联动中枢", "Linkage Hub"],
    ["未来决策舱", "Future Decision Deck"], ["接口文档", "API Documentation"],
    ["能耗与碳排 指挥盘", "Energy & Carbon Command Board"], ["预警", "Alerts"],
    ["预测校准 · 近窗", "Forecast Calibration · Near Window"],
    ["越峰风险（15/30/60 分钟）", "Peak-Risk Outlook · 15/30/60 min"],
    ["将发生什么（未来 60 分钟）", "What Happens Next · 60 min"],
    ["联动与导出", "Linkage & Export"],
    ["港区动态渲染（AGV / 岸桥 / 场桥 / 拖车 / 泊位）", "Live Port Rendering · AGV / QC / YC / Trucks / Berths"],
    ["控制侧栏（QC-01）", "Control Sidebar · QC-01"],
    ["策略编排 & 闭环执行（预测 / RL / 执行 / 闭环）", "Strategy Orchestration & Closed-Loop Execution"],
    ["策略出现原因 → 仿真支撑 → 下发判断", "Strategy Rationale → Simulation Evidence → Dispatch Decision"],
    ["聚合仿真（总负荷 P10/P50/P90）", "Aggregate Simulation · Total Load P10/P50/P90"],
    ["RL 策略面板 · 集成", "Integrated RL Strategy Panel"], ["输出", "Outputs"],
    ["可解释特征（SHAP 简版）", "Explainable Features · SHAP Summary"],
    ["策略下发（演示）", "Strategy Dispatch · Demo"],
    ["孪生与可靠性（Twin Fidelity · 场景压测 · 重放）", "Twin Reliability · Fidelity / Stress Test / Replay"],
    ["RL Ops Center（OPE｜守护栏｜可观测性｜实验｜因果）", "RL Ops Center · OPE / Guardrails / Observability / Experiments / Causality"],
    ["执行与闭环（审批 / 一键下发 + A/B + 在线学习）", "Execution Loop · Approval / Dispatch / A-B / Online Learning"],
    ["一键操作", "Quick Actions"], ["在线设备快照", "Live Asset Snapshot"], ["冲突与调解", "Conflicts & Mediation"],
    ["上线与运维控制（OpsX）", "Release & Operations Control · OpsX"],
    ["对接外部环境（TOS/WMS · 电力市场 · AIS · 潮汐）", "External Integrations · TOS/WMS / Power Market / AIS / Tides"],
    ["三、运营级（MLOps/GRC/集成）", "3. Operations Layer · MLOps / GRC / Integration"],
    ["四、港口业务深耕（价值拉满）", "4. Port Operations Intelligence · Value Layer"],
    ["5. 报表与合规", "5. Reporting & Compliance"],
    ["TwinLab（场景工厂｜韧性演练｜数据契约）", "TwinLab · Scenario Factory / Resilience / Data Contracts"],
    ["监测与运维 · 异常检测 / 漂移", "Monitoring & Operations · Anomaly / Drift"],
    ["异常结果", "Anomaly Results"], ["PSI 漂移", "PSI Drift"],
    ["模块A · 训练与评估（AGV 充/换电）", "Module A · Training & Evaluation · AGV Charging"],
    ["模块B · 堆场照明（Yard Lighting）训练/评估快照", "Module B · Yard Lighting Training / Evaluation"],
    ["模块C · 制冷（HVAC Cooling）输出数据", "Module C · HVAC Cooling Outputs"],
    ["模块D · 岸电储能（Shore BESS）输出数据", "Module D · Shore Power & BESS Outputs"],
    ["模块E · 场内储能（BESS Energy）输出数据", "Module E · BESS Energy Outputs"],
    ["模块F · 堆场吊机（Yard Crane）输出数据", "Module F · Yard Crane Outputs"],
    ["模块G · 岸桥（QC）输出数据（Port G QC MVP）", "Module G · Quay Crane Outputs · Port G QC MVP"],
    ["项目联动中枢", "Project Linkage Hub"], ["回首页", "Back to Home"],
    ["打开 RL 面板", "Open RL Panel"], ["刷新联动状态", "Refresh Linkage Status"],
    ["小懿问答 / 指令网关", "Xiaoyi Q&A · Command Gateway"], ["启动小懿AI", "Start Xiaoyi AI"],
    ["让小懿判断", "Ask Xiaoyi"], ["确认执行", "Confirm Execution"], ["清空", "Clear"],
    ["RL 训练 / 测试链路", "RL Training & Test Pipeline"], ["低风险训练", "Low-Risk Training"],
    ["查看训练状态", "View Training Status"], ["策略测试", "Policy Test"], ["进入训练 UI", "Open Training UI"],
    ["航行模拟器状态", "Sailing Simulator Status"], ["打开 Godot 航行模拟器", "Open Godot Sailing Simulator"],
    ["启动航线演示", "Start Route Demo"], ["切换船舶视角", "Switch Vessel View"],
    ["运行 smoke test", "Run Smoke Test"], ["动作日志", "Action Log"],
    ["同步航行日志", "Sync Sailing Logs"], ["清空前端日志", "Clear Frontend Logs"],
    ["生成 Copilot 答案", "Generate Copilot Answer"], ["重置", "Reset"],
    ["快捷 Playbook", "Quick Playbooks"], ["复制审计包", "Copy Audit Packet"],
    ["回主平台策略区", "Back to Strategy"], ["回主平台首页", "Back to Home"],
    ["刷新接入口 / 健康检查", "Refresh Connectors · Health Check"], ["启动训练", "Start Training"],
    ["暂停", "Pause"], ["查看状态", "View Status"], ["拉取策略列表", "Load Policies"],
    ["先模拟", "Simulate First"], ["验证上线(dry-run)", "Validate Release · Dry Run"],
    ["仿真通过后记录 dry-run 下发", "Record Dry-Run Dispatch After Simulation"],
    ["刷新下发历史", "Refresh Dispatch History"], ["小懿请求执行 RL 训练", "Xiaoyi Requests RL Training"],
    ["取消", "Cancel"], ["开始执行", "Start Execution"],
    ["您好！我是小懿AI", "Hello! I’m Xiaoyi AI"], ["您的港航智能助手", "Your Smart Maritime Assistant"],
    ["正常", "Healthy"], ["在线", "Online"], ["离线", "Offline"], ["待命", "Standby"],
    ["运行中", "Running"], ["暂停中", "Paused"], ["已完成", "Completed"], ["待审批", "Pending Approval"],
    ["刷新", "Refresh"], ["加载", "Load"], ["查询", "Query"], ["执行", "Execute"],
    ["确认", "Confirm"], ["关闭", "Close"], ["保存", "Save"], ["导出", "Export"], ["复制", "Copy"],
    ["推荐", "Recommend"], ["仿真", "Simulate"], ["审批", "Approve"], ["回执", "Receipt"], ["审计", "Audit"],
    ["削峰", "Peak Shaving"], ["降碳", "Carbon Reduction"], ["保供", "Reliability"], ["成本", "Cost"]
  ]);

  const selectorEntries = [
    ["#home-lead", "Port-wide operating conclusion and recommended next action", "block"],
    ["#home-desc", "Decision-first overview linking twin simulation, strategy, execution, and audit", "block"],
    ["#home-focus", "Recommended Next Action", "block"],
    ["#home-status-sub", "Live operating health and execution-consistency summary", "block"],
    ["#home-risk-sub", "Forward risk signals across load, approval, alerts, and linkage", "block"],
    ["#home-focus-sub", "Recommended next step based on current platform evidence", "block"],
    [".contract-registry-sub", "Contracts are presented as the data foundation of the platform", "block"],
    ["#home-loop-strategy", "Strategy judgment and confidence evidence", "block"],
    ["#home-loop-dispatch", "Approval, dispatch, and receipt-loop evidence", "block"],
    ["#home-loop-audit", "Governance, audit, and contract-readiness evidence", "block"],
    ["#ctx-status-card-sub", "Overview status and risk context carried into the twin", "block"],
    ["#ctx-scene-card-sub", "Scenario context propagated to simulation and strategy", "block"],
    ["#ctx-forecast-card-sub", "Six-hour forecast evidence for strategy selection", "block"],
    ["#twin-chain-sub", "Overview context is carried into the active twin scenario", "block"],
    ["#strategy-origin-sub", "Traceable source context for the current strategy", "block"]
  ];

  const placeholderMap = new Map([
    ["例如：小懿，开始训练碳排最低目标", "例如：小懿，开始训练碳排最低目标 / e.g. Xiaoyi, train for minimum carbon"],
    ["输入港航问题或操作指令", "输入港航问题或操作指令 / Enter a maritime question or command"]
  ]);

  function createStyle(){
    if (document.getElementById("port-bilingual-ui-style")) return;
    const style = document.createElement("style");
    style.id = "port-bilingual-ui-style";
    style.textContent = `
      html{--i18n-en:#86a9d6;--i18n-en-strong:#8edcf6}
      [data-i18n-en]{--i18n-content:attr(data-i18n-en)}
      [data-i18n-en].i18n-en-block::after{
        content:var(--i18n-content);display:block;margin-top:3px;color:var(--i18n-en);font-size:max(9px,.58em);font-weight:650;line-height:1.28;letter-spacing:.035em;text-transform:none;white-space:normal;opacity:.92
      }
      [data-i18n-en].i18n-en-inline::after{
        content:" / " var(--i18n-content);color:var(--i18n-en);font-size:max(8px,.68em);font-weight:650;line-height:inherit;letter-spacing:.025em;text-transform:none;white-space:normal;opacity:.92
      }
      h1[data-i18n-en].i18n-en-block::after,h2[data-i18n-en].i18n-en-block::after,h3[data-i18n-en].i18n-en-block::after,h4[data-i18n-en].i18n-en-block::after,
      .title[data-i18n-en].i18n-en-block::after,.home-lead[data-i18n-en].i18n-en-block::after,.hero-title[data-i18n-en].i18n-en-block::after{color:var(--i18n-en-strong);letter-spacing:.045em}
      button[data-i18n-en]::after,a[data-i18n-en]::after{pointer-events:none}
      .nav-dropdown [data-i18n-en].i18n-en-block::after{margin-top:1px;font-size:8px;line-height:1.15;color:#7597c7}
      .triad-value[data-i18n-en].i18n-en-block::after,.kpi .v[data-i18n-en].i18n-en-block::after{font-size:9px;color:#7dd3fc}
      .xiaoyi-sprite-speech [data-i18n-en].i18n-en-block::after{font-size:9px;margin-top:1px;color:#82c9ef}
      @media(max-width:860px){[data-i18n-en].i18n-en-inline::after{display:block;margin-top:2px}.nav-shell [data-i18n-en].i18n-en-inline::after{display:inline;margin-top:0}}
    `;
    document.head.appendChild(style);
  }

  function modeFor(el){
    if (el.matches("button,a,label,option,.chip,.badge,.nav-cn,.small")) return "inline";
    if (el.matches("h1,h2,h3,h4,h5,h6,.title,.subtitle,.home-lead,.home-desc,.triad-label,.triad-value,.triad-sub,.dash-title,.panel-title,.mini-title,.app-title,.pm-layer-title,.contract-name,.contract-meta,.contract-detail-title,.contract-detail-body")) return "block";
    return el.textContent && el.textContent.trim().length > 18 ? "block" : "inline";
  }

  function setEnglish(el, english, mode, source){
    if (!el || !english) return;
    if (el.matches(TITLE_LIKE_SELECTOR)) {
      clearEnglish(el);
      return;
    }
    el.dataset.i18nEn = english;
    el.dataset.i18nSource = source || "exact";
    el.classList.toggle("i18n-en-block", mode === "block");
    el.classList.toggle("i18n-en-inline", mode !== "block");
  }

  function clearEnglish(el){
    if (!el) return;
    delete el.dataset.i18nEn;
    delete el.dataset.i18nSource;
    el.classList.remove("i18n-en-block","i18n-en-inline");
  }

  function dynamicTranslation(text){
    if (/^当前选中：/.test(text)) return "Currently Selected Contract";
    if (/^规则口径：/.test(text)) return "Rule-based operating assessment";
    if (/^峰值需量风险=/.test(text)) return "Peak-demand and pending-approval risk evidence";
    if (/^待审批=/.test(text)) return "Pending approvals and latest execution receipt";
    if (/^策略可信度：/.test(text)) return "AI strategy confidence and priority-action evidence";
    if (/^当前优先动作：/.test(text)) return "Current priority action and recommended workflow";
    if (/^先看孪生推演/.test(text)) return "Review Twin Simulation First";
    if (/^先看主链路巡检/.test(text)) return "Review Main Workflow First";
    if (/^先看/.test(text)) return "Review Recommended Workflow First";
    if (/^OpsX 当前摘要：/.test(text)) return "Current OpsX governance summary";
    if (/^Contract Registry：/.test(text)) return "Contract readiness and real-adapter targets";
    if (/^当前主判断为/.test(text)) return "Current overview status, risk, and operating focus";
    if (/^该场景由 Overview/.test(text)) return "Scenario context inherited from Overview";
    if (/^当前从 Overview/.test(text)) return "Overview context carried into Twin and Strategy";
    if (/^当前策略来源：/.test(text)) return "Traceable source context for the active strategy";
    return "";
  }

  function shouldSkip(el){
    return !el || el.nodeType !== 1 || el.matches("script,style,noscript,code,pre,svg,path,input,textarea,select") || Boolean(el.closest(".nav-cn,.nav-en,.en,[data-i18n-skip]"));
  }

  function applyExact(el){
    if (shouldSkip(el) || el.children.length) return;
    const text = (el.textContent || "").replace(/\s+/g," ").trim();
    if (!text) return;
    const english = exact.get(text) || dynamicTranslation(text);
    if (english) setEnglish(el, english, modeFor(el), "exact");
    else if (el.dataset.i18nSource === "exact") {
      delete el.dataset.i18nEn;
      delete el.dataset.i18nSource;
      el.classList.remove("i18n-en-block","i18n-en-inline");
    }
  }

  function applyTree(root){
    const base = root && root.nodeType === 1 ? root : document.body;
    if (!base) return;
    applyExact(base);
    base.querySelectorAll("h1,h2,h3,h4,h5,h6,button,a,label,span,div,p,small,strong,b").forEach(applyExact);
    document.querySelectorAll("input[placeholder],textarea[placeholder]").forEach(el=>{
      const next = placeholderMap.get(el.getAttribute("placeholder") || "");
      if (next) el.setAttribute("placeholder", next);
    });
    selectorEntries.forEach(([selector, english, mode])=>{
      document.querySelectorAll(selector).forEach(el=>setEnglish(el, english, mode, "selector"));
    });
    document.querySelectorAll(TITLE_LIKE_SELECTOR).forEach(clearEnglish);
    document.querySelectorAll(".nav-cn[data-i18n-en]").forEach(el=>{
      clearEnglish(el);
    });
    document.documentElement.dataset.bilingualMode = "zh-primary-en-assist";
  }

  let queued = false;
  function schedule(root){
    if (queued) return;
    queued = true;
    requestAnimationFrame(()=>{
      queued = false;
      applyTree(root || document.body);
    });
  }

  function install(){
    createStyle();
    document.title = `${document.title.replace(/\s*\|\s*Port AI Operations Platform$/i,"")} | Port AI Operations Platform`;
    applyTree(document.body);
    const observer = new MutationObserver(mutations=>{
      const target = mutations.find(m=>m.target && m.target.nodeType === 1)?.target || document.body;
      schedule(target);
    });
    observer.observe(document.body,{subtree:true,childList:true,characterData:true});
    window.__refreshBilingualUi = ()=>applyTree(document.body);
    window.__portBilingualUiVersion = VERSION;
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install, {once:true});
  else install();
})();
