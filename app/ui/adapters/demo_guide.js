(function(){
  "use strict";
  if(window.__portDtDemoGuideInstalled) return;
  window.__portDtDemoGuideInstalled = true;

  const evidenceScript = document.createElement("script");
  evidenceScript.src = "/static/demo-evidence.js?v=20261021";
  document.head.append(evidenceScript);

  const STORE_KEY = "port_dt_native_demo_guide_v1";
  const VERSION = "2026-10-31-click-audit-v4";
  const RECEIPT_EVENT = "port-dt:demo-receipt";
  const DEFAULT_STATE = {
    open:false, collapsed:false, current:0, phase:"ready", completed:[], skipped:[],
    statusText:"点击当前真实操作开始演示。", pendingExecute:"", updatedAt:""
  };

  const STEPS = [
    {id:"trained-evidence",route:"/",phase:"00 场景证据",label:"显示全部已训练场景",title:"已训练场景算法与数据清单",location:"训练证据清单",selector:"[data-demo-evidence-open]",region:".demo-evidence-panel",result:".demo-evidence-panel",speech:"每个已训练场景列出具体算法、真实和工程数据集的名称与数量、State、Reward、Action和输出，另附训练记录、模型哈希和准入状态。移动端使用相同训练服务与证据口径。",wait:"清单读取成功后显示全部场景，可逐项切换核验。",timeoutMs:10000},
    {
      id:"opening", route:"/", phase:"01 业务总览", label:"查看管理驾驶舱",
      title:"港口数字孪生解决什么问题", location:"首页 · 管理驾驶舱",
      selector:'a.home-action[data-panel="cockpit"]', region:"#exec-cockpit", result:"#exec-cockpit",
      speech:"港口计划、设备、能源与安全数据通常分散。系统用公开数据校准的数字孪生统一态势，用离线 RL 比较策略，再由舷鸥编排任务、人工确认、安全门和审计凭证形成闭环。当前展示不是现场生产控制。",
      wait:"管理驾驶舱进入视野，状态等级、峰值风险、可信度与收益摘要同时高亮。", timeoutMs:5000
    },
    {
      id:"situation", route:"/", phase:"02 数据与实时态势", label:"播放态势回放",
      title:"同一时间轴上的运行态势", location:"数字孪生 · 态势故事",
      selector:"#story-play", reveal:'a[href="#story-section"][data-module-page]', region:"#story-section", result:"#story-section",
      speech:"播放公开数据校准的连续回放。时间、事件、负荷、成本、碳排和安全门会同步变化，让观众直接看到基线与候选策略的差异。",
      wait:"时间轴开始滚动，播放状态、图表与事件标记持续更新。", timeoutMs:10000
    },
    {
      id:"dataset", route:"/", phase:"03 官方来源与血缘", label:"查看公开来源与哈希",
      title:"官方数据来源、版本与边界", location:"外部信号 · 数据血缘",
      selector:"#btn-ext-provenance", reveal:'a[href="#ext-section"][data-module-page]', region:"#ext-section", result:"#ext-v3-detail",
      speech:"先核对来源机构、官方地址、许可说明、时间范围、字段、版本和 SHA-256。官方原始观测、派生字段与工程模拟分层标识，未接入的现场接口不会显示为在线。",
      wait:"数据血缘详情展示来源、版本、适配状态与校验标识。", timeoutMs:15000
    },
    {
      id:"rl-dataset", route:"/rl-panel", phase:"04 RL 训练数据集", label:"聚焦训练数据集",
      title:"哪些数据真正进入训练", location:"RL 面板 · 训练中心",
      selector:"[data-proof-datasets-focus]", region:"#rlProofConsole", result:"[data-proof-dataset-panel]",
      speech:"训练中心单独列出可训练数据包。这里看记录量、独立官方观测、证据等级、因素覆盖、场景包和 SHA-256；行数不等于独立信息量，公开校准包也不冒充码头生产遥测。",
      wait:"数据集对照表获得明显聚焦，关键数据行重新进入视野。", timeoutMs:18000
    },
    {
      id:"rl-contract", route:"/rl-panel", phase:"05 RL 训练契约", label:"展开训练契约",
      title:"State、Action、Reward 与安全门", location:"RL 面板 · 训练中心",
      selector:"[data-proof-contract-focus]", region:"#rlProofConsole", result:"[data-proof-contract-panel]",
      speech:"宽屏契约展示状态维度、连续动作、Reward 权重、算法实现和验证门槛。动作只生成离线建议，不包含执法、放行或现场写控制权。",
      wait:"目标函数、观测、动作与因素标签面板被聚焦，算法证据保持可见。", timeoutMs:18000
    },
    {
      id:"xiaoyi-command", route:"/rl-panel", phase:"06 舷鸥下发训练任务", label:"让舷鸥理解并下发",
      title:"自然语言变成受控 RL 任务", location:"舷鸥悬浮助手 · 真实指令网关",
      selector:'.xiaoyi-sprite-btn[type="submit"]', region:".xiaoyi-sprite-root", result:".xiaoyi-sprite-log",
      speech:"输入演示级快速训练目标。舷鸥识别成本与碳排平衡、严格安全边界，生成 PPO 参数和 320 步链路验证任务；它只能把任务送入人工确认门，不能绕过确认取得生产控制权。",
      wait:"舷鸥逐步显示目标识别、推荐算法、参数与权限边界，然后在同一窗口进入 RL 面板。", timeoutMs:20000
    },
    {
      id:"training-review", route:"/rl-panel", phase:"07 人工复核训练参数", label:"打开训练确认门",
      title:"目标、参数和风险由人确认", location:"RL 面板 · 训练配置",
      selector:"#btnStartTrain", region:".train-card", result:"#assistantConfirmBackdrop",
      speech:"现在复核数据集、算法、训练步数、Reward 权重和安全风险。点击只打开确认门，尚未调用训练接口；这一步证明舷鸥没有越过人工决策。",
      wait:"人工确认框显示完整配置快照、风险说明和开始执行按钮。", timeoutMs:8000
    },
    {
      id:"training-start", route:"/rl-panel", phase:"08 启动真实演示训练", label:"确认并启动真实训练",
      title:"立即响应、进度可见、失败可恢复", location:"RL 面板 · 人工确认门",
      selector:"#btnConfirmAssistantRun", region:"#assistantConfirmBackdrop", result:"#trainJobChip",
      speech:"确认后才调用真实训练接口。演示档为 320 步 PPO，只用于展示任务受理、奖励、损失、进度和检查点链路，不能据此宣称收敛或生产收益。",
      wait:"确认框关闭，任务编号、状态、进度条、Reward 与日志立即开始更新。", timeoutMs:30000
    },
    {
      id:"training-status", route:"/rl-panel", phase:"09 训练进度与回执", label:"查看训练状态",
      title:"从后端回收可审计训练结果", location:"RL 面板 · 训练运行态",
      selector:"#btnPollTrainStatus", region:"#trainRunSummary", result:"#trainRunSummary",
      speech:"主动读取真实后端状态，核对 step、Reward、损失、策略版本、检查点和失败原因。训练未完成也不会锁死演示，可重试、重置或保底继续。",
      wait:"运行摘要和日志刷新为当前真实状态，并显示任务进度或终态。", timeoutMs:12000
    },
    {
      id:"convergence", route:"/", phase:"10 历史收敛与训练证据", label:"查看优化/收敛判据",
      title:"奖励、损失和多种子验证", location:"模块 D · 岸电储能",
      selector:"#btn-d-convergence", reveal:'a[href="#sbess-section"][data-module-page]', region:"#sbess-section", result:"#d-evidence-detail",
      speech:"历史正式实验与刚才的 320 步烟测分开。这里看 Reward、Actor/Critic 或对应损失、成功率、多种子验证、检查点和选择门禁，失败与负向结果同样保留。",
      wait:"历史曲线和关键点进入视野，证据详情切换到收敛与训练过程。", timeoutMs:22000
    },
    {
      id:"business", route:"/", phase:"11 已训练业务指标", label:"查看配对业务评估",
      title:"基线、单目标与综合策略对比", location:"模块 D · 岸电储能",
      selector:"#btn-d-business", reveal:'a[href="#sbess-section"][data-module-page]', region:"#sbess-section", result:"#d-evidence-detail",
      speech:"在相同数据指纹和窗口下比较成本、碳排、峰值与约束结果，并给出模型版本、检查点和适用范围。负改善如实显示，年化值只是公开工程场景外推。",
      wait:"业务指标、盲测协议、质量门和结果边界替换进入详情区。", timeoutMs:22000
    },
    {
      id:"safety", route:"/", phase:"12 验证与安全门", label:"查看现场准入与声明边界",
      title:"训练完成不等于直接上线", location:"AI 可信度 · 现场准入",
      selector:"#btn-trust-boundary", reveal:'a[href="#ai-trust-scenes"][data-module-page]', region:"#ai-trust-scenes", result:"#trust-evidence-detail",
      speech:"模型还要经过独立回放、约束验证、漂移检查、人工确认、影子运行、灰度和回滚。页面明确显示生产权限状态，避免把离线验证写成现场授权。",
      wait:"准入阶梯、声明注册表和生产权限边界进入详情区。", timeoutMs:18000
    },
    {
      id:"audit", route:"/", phase:"13 审计回溯", label:"查看历史审计哈希",
      title:"任务、参数、结果与追踪凭证", location:"OpsX · 审计证据",
      selector:"#btn-opsx-audit", reveal:'a[href="#opsx-section"][data-module-page]', region:"#opsx-section", result:"#opsx-v3-detail",
      speech:"最后回看任务目标、目标系统、数据版本、算法参数、执行状态、业务指标、验证结论、Trace ID、时间戳和生产权限。现场遥测和南向写控制仍保持关闭。",
      wait:"审计详情展示可追踪 manifest 与历史哈希，数字孪生闭环结束。", timeoutMs:18000
    },
    {
      id:"twin3d-showcase", route:"/", group:"quick", phase:"14 能力速览 · 高端可视化", label:"打开 Twin 3D 大屏",
      title:"三态联动的港区数字孪生大屏", location:"数字孪生 · Twin 3D",
      selector:'a[href="#twin3d-section"][data-module-page]', triggerSelector:'.nav-group[data-menu-group="twin"] > .nav-trigger', revealMenu:'.nav-group[data-menu-group="twin"] > .nav-trigger',
      region:"#twin3d-section", result:"#twinModeConsole",
      speech:"核心闭环讲完后，快速看系统的高端可视化能力。三维港区把当前态、未来六小时和策略作用后放在同一屏，并联动能耗热力、碳排图层、设备曲线与风险窗口。它是公开数据校准视图，不冒充现场实况。",
      wait:"页面切换到 Twin 3D，三态轨道、指标图层、三维场景和联动曲线进入视野。", timeoutMs:20000
    },
    {
      id:"multiport-showcase", route:"/", group:"quick", phase:"15 能力速览 · 多港复用", label:"打开多港区总览",
      title:"同一能力契约适配多个港区", location:"多港区 · 复用总览",
      selector:'a[href="#multiport-section"][data-module-page]', triggerSelector:'.nav-group[data-menu-group="multiport"] > .nav-trigger', revealMenu:'.nav-group[data-menu-group="multiport"] > .nav-trigger',
      region:"#multiport-section", result:"#multiport-section",
      speech:"多港区页面说明算法、数据和接口如何按港区配置复用。共享的是状态、动作、安全门和审计契约；不同港口的数据映射、权重和现场权限必须分别校准。",
      wait:"多港区卡片、数据覆盖和复用边界成为当前独立页面。", timeoutMs:8000
    },
    {
      id:"esg-showcase", route:"/", group:"quick", phase:"16 能力速览 · ESG 合规", label:"打开 ESG 总览",
      title:"能耗、碳排与合规口径同屏", location:"ESG / 合规 · 总览",
      selector:'a[href="#esg-section"][data-module-page]', triggerSelector:'.nav-group[data-menu-group="esg"] > .nav-trigger', revealMenu:'.nav-group[data-menu-group="esg"] > .nav-trigger',
      region:"#esg-section", result:"#esg-section",
      speech:"ESG 页面把能耗、碳排、单位吞吐指标和报表口径放在一起，支持按泊位、堆场和业务单元追溯。这里展示的是核算与模拟证据，不是未经审计的生产减排声明。",
      wait:"ESG 指标、分摊口径和合规入口切换为当前页面。", timeoutMs:8000
    },
    {
      id:"strategy-showcase", route:"/", group:"quick", phase:"17 能力速览 · 策略执行", label:"打开策略编排总览",
      title:"从候选策略到执行前安全门", location:"策略 → 执行 · 编排总览",
      selector:'a[href="#strategy-exec-module"][data-module-page]', triggerSelector:'.nav-group[data-menu-group="strategy"] > .nav-trigger', revealMenu:'.nav-group[data-menu-group="strategy"] > .nav-trigger',
      region:"#strategy-exec-module", result:"#strategy-exec-module",
      speech:"策略编排把预测、RL 候选、多智能体协同、TwinLab 场景和执行前审批串成一条受控链路。此处只做快速能力展示，真正的训练证据和人工门已经在前面的核心步骤中完成。",
      wait:"策略链路、候选方案和执行边界切换为当前独立页面。", timeoutMs:8000
    },
    {
      id:"ops-copilot-showcase", route:"/ops-copilot", sourceSelector:'a[data-route="/ops-copilot"]', group:"quick", landingReceipt:true, phase:"18 能力速览 · 运营副驾", label:"进入运营副驾",
      title:"舷鸥面向一线运行的证据化问答", location:"运营副驾 · Ops Copilot",
      sourceRegion:"#panel-nav-primary", region:".hero", result:"#missionRail",
      speech:"运营副驾面向值班、能源、设备与审计角色，把当前态势、未来风险、SOP、证据来源和人工交接合成可执行建议。调用模型、上下文哈希和权限边界都可见。",
      wait:"同一窗口进入运营副驾，任务卡、运行证据条和舷鸥工作台完整显示。", timeoutMs:12000
    },
    {
      id:"v3-showcase", route:"/v3", sourceSelector:'a[data-route="/v3?from=home"]', group:"quick", landingReceipt:true, phase:"19 能力速览 · 决策中枢", label:"进入 V3 决策中枢",
      title:"面向决策层的证据驾驶舱", location:"V3 决策中枢",
      sourceRegion:"#panel-nav-primary", region:"#top", result:".metric-row",
      speech:"V3 决策中枢以更高层的视觉语言汇总多港数据、算法覆盖、盲测证据、业务价值和部署缺口。所有优势结论都绑定数据哈希、切分方式与安全边界。",
      wait:"同一窗口进入 V3，高端港区态势图、证据指标和策略价值工作台显示。", timeoutMs:12000
    },
    {
      id:"integration-showcase", route:"/integration-hub", sourceSelector:'a[data-route="/integration-hub"]', group:"quick", landingReceipt:true, phase:"20 能力速览 · 项目联动", label:"进入项目联动中枢",
      title:"舷鸥、RL 与外部系统的受控联动", location:"项目联动中枢",
      sourceRegion:"#panel-nav-primary", region:".impact-deck", result:".status-strip",
      speech:"项目联动中枢集中展示舷鸥任务编排、RL 策略验证、外部系统健康状态和统一动作日志。系统离线会明确显示，任何执行都保留人工确认和不具备生产控制权的边界。",
      wait:"同一窗口进入联动中枢，系统状态、策略验证飞行甲板和统一日志显示。", timeoutMs:12000
    },
    {
      id:"future-showcase", route:"/rl_future/rl_future_panel.html", sourceSelector:'a[data-route="/rl_future/rl_future_panel.html"]', group:"quick", landingReceipt:true, phase:"21 能力速览 · 未来决策", label:"进入未来决策舱",
      title:"反事实候选与安全护栏的未来舱", location:"未来决策舱",
      sourceRegion:"#panel-nav-primary", region:".hero-grid", result:".core-panel",
      speech:"未来决策舱用于快速展示反事实候选、正式盲测参照、守护栏矩阵和审计回执。视觉效果更强，但仍然只运行离线证据推演，不向生产设备下发指令。",
      wait:"同一窗口进入未来舱，全息态势、候选策略与安全护栏矩阵显示。", timeoutMs:12000
    },
    {
      id:"api-docs-showcase", route:"/docs", sourceSelector:'a[data-route="/docs"]', group:"quick", landingReceipt:true, phase:"22 能力速览 · 开放接口", label:"打开 API 接口文档",
      title:"可集成、可替换的标准化接口", location:"接口文档 · OpenAPI",
      sourceRegion:"#panel-nav-primary", region:"#swagger-ui", result:"#swagger-ui",
      speech:"最后展示系统提供的标准化接口。OpenAPI 文档列出数据、数字孪生、RL、验证、审计和联动端点，便于接入 TOS、ECS、EMS 与上层应用；接口存在不代表已获得现场写控制权限。",
      wait:"同一窗口打开真实 OpenAPI 文档，接口分组、请求模型和可机读契约显示。", timeoutMs:12000
    }
  ];

  let state = loadState();
  let root = null;
  let launcher = null;
  let receiptTimer = null;
  let waitingTrainingContract = false;
  let programmaticClick = false;
  let actionBusy = false;

  function loadState(){
    try{
      const saved = JSON.parse(sessionStorage.getItem(STORE_KEY) || "null");
      const merged = {...DEFAULT_STATE,...(saved && typeof saved === "object" ? saved : {})};
      if(merged.current < STEPS.length && merged.phase === "finished"){
        merged.phase = merged.completed.includes(STEPS[merged.current]?.id) ? "done" : "ready";
        merged.statusText = "已恢复演示进度；完成情况以各步骤实际回执为准。";
      }
      return merged;
    }catch(_){ return {...DEFAULT_STATE}; }
  }
  function saveState(){ state.updatedAt = new Date().toISOString(); sessionStorage.setItem(STORE_KEY, JSON.stringify(state)); }
  function currentStep(){ return STEPS[state.current] || null; }
  function pathIs(path){ return window.location.pathname === path; }
  function isVisible(node){
    if(!node) return false;
    const style = window.getComputedStyle(node);
    return style.display !== "none" && style.visibility !== "hidden" && node.getClientRects().length > 0;
  }
  function findTarget(selector){
    const nodes = Array.from(document.querySelectorAll(selector));
    return nodes.find(node => !node.disabled && isVisible(node)) || nodes.find(node => !node.disabled) || nodes[0] || null;
  }
  function escapeHtml(value){
    return String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));
  }

  function addStyle(){
    if(document.getElementById("port-dt-demo-guide-style")) return;
    const style = document.createElement("style");
    style.id = "port-dt-demo-guide-style";
    style.textContent = `
      body.port-demo-open{padding-left:306px!important;transition:padding-left .18s ease}body.port-demo-collapsed{padding-left:68px!important}
      .port-demo-guide{position:fixed;z-index:12050;left:10px;top:10px;bottom:10px;width:286px;display:flex;flex-direction:column;overflow:hidden;border:1px solid rgba(56,189,248,.48);border-radius:15px;background:linear-gradient(180deg,rgba(5,18,36,.985),rgba(4,13,27,.985));box-shadow:0 22px 58px rgba(0,0,0,.46),inset 0 1px 0 rgba(255,255,255,.05);color:#eaf5ff;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,"Noto Sans SC",sans-serif}
      .port-demo-guide[hidden],.port-demo-launcher[hidden]{display:none!important}.port-demo-guide[data-collapsed="true"]{width:50px;height:50px;bottom:auto}.port-demo-guide[data-collapsed="true"] .pdg-wide{display:none}.port-demo-guide[data-collapsed="true"] .pdg-head{padding:9px 6px;justify-content:center}.port-demo-guide[data-collapsed="true"] .pdg-close{display:none}.port-demo-guide[data-collapsed="true"] .pdg-collapse{transform:rotate(180deg)}
      .pdg-head{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;padding:11px;border-bottom:1px solid rgba(96,165,250,.18);background:linear-gradient(110deg,rgba(15,64,105,.72),rgba(10,35,65,.32))}.pdg-kicker{font-size:8px;letter-spacing:.14em;color:#67e8f9;font-weight:900}.pdg-title{margin-top:3px;font-size:14px;font-weight:950}.pdg-head-actions{display:flex;gap:4px}.pdg-icon{width:28px;height:28px;padding:0;border:1px solid rgba(148,163,184,.28);border-radius:8px;background:#0a1930;color:#cfe8ff;font-weight:900;cursor:pointer}
      .pdg-body{min-height:0;overflow:auto;padding:10px;overscroll-behavior:contain}.pdg-progress-meta{display:flex;justify-content:space-between;color:#8db1d5;font-size:9px}.pdg-progress{height:5px;margin:6px 0 9px;border-radius:99px;background:#07111f;overflow:hidden}.pdg-progress i{display:block;height:100%;background:linear-gradient(90deg,#38bdf8,#2dd4bf,#fbbf24);transition:width .24s ease;box-shadow:0 0 12px rgba(45,212,191,.42)}
      .pdg-steps{display:grid;gap:4px;margin:0 0 9px;padding:0;list-style:none}.pdg-step{display:grid;grid-template-columns:20px 1fr auto;align-items:center;gap:6px;min-height:31px;padding:4px 6px;border:1px solid rgba(96,165,250,.13);border-radius:8px;background:rgba(7,20,38,.72);color:#7797b9;font-size:9px;cursor:pointer}.pdg-step:hover{border-color:rgba(56,189,248,.45);color:#d7ecff}.pdg-step i{display:grid;width:18px;height:18px;place-items:center;border:1px solid rgba(96,165,250,.25);border-radius:50%;font-style:normal;font-weight:900}.pdg-step b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pdg-step.active{border-color:rgba(34,211,238,.68);background:rgba(8,58,82,.72);color:#f0fbff}.pdg-step.active i{border-color:#22d3ee;color:#67e8f9;box-shadow:0 0 10px rgba(34,211,238,.24)}.pdg-step.done{color:#9be8c5;border-color:rgba(52,211,153,.24)}.pdg-step.skipped{color:#d8ba78}.pdg-step em{font-style:normal;font-size:8px}
      .pdg-section-divider{margin:6px 0 2px;padding:6px 2px 2px;border-top:1px solid rgba(45,212,191,.28);color:#5eead4;font-size:8px;font-weight:950;letter-spacing:.12em;text-transform:uppercase}
      .pdg-phase{color:#67e8f9;font-size:8px;font-weight:900;letter-spacing:.08em}.pdg-step-title{margin-top:3px;font-size:14px;font-weight:950;line-height:1.35;color:#f7fbff}.pdg-location{margin-top:3px;color:#7ea4c9;font-size:9px}.pdg-speech{margin-top:8px;padding:9px 10px;border-left:3px solid #38bdf8;border-radius:0 9px 9px 0;background:rgba(14,45,77,.56);font-size:11px;line-height:1.62;color:#cfe5fa}.pdg-wait{margin-top:7px;color:#91b2d2;font-size:9px;line-height:1.5}.pdg-status{margin-top:8px;padding:7px 8px;border:1px solid rgba(96,165,250,.18);border-radius:8px;background:#071528;color:#9ebddd;font-size:10px;line-height:1.45}.pdg-status[data-state="waiting"]{color:#fde68a;border-color:rgba(245,158,11,.38)}.pdg-status[data-state="done"],.pdg-status[data-state="finished"]{color:#a7f3d0;border-color:rgba(52,211,153,.38)}.pdg-status[data-state="error"]{color:#fecaca;border-color:rgba(248,113,113,.42)}
      .pdg-primary{width:100%;min-height:42px;margin-top:9px;padding:7px 9px;border:1px solid rgba(34,211,238,.58);border-radius:10px;background:linear-gradient(135deg,#0369a1,#0e7490);color:#fff;font-size:11px;font-weight:950;cursor:pointer;box-shadow:0 8px 22px rgba(6,182,212,.14)}.pdg-primary.is-next{background:linear-gradient(135deg,#047857,#0f766e);border-color:#34d399}.pdg-primary:disabled{opacity:.58;cursor:not-allowed}.pdg-controls{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:7px}.pdg-controls button{min-height:31px;border:1px solid rgba(96,165,250,.22);border-radius:8px;background:#0d2038;color:#cfe4fb;font-size:9px;font-weight:850;cursor:pointer}.pdg-controls .pdg-fallback{color:#fde68a;border-color:rgba(245,158,11,.32)}.pdg-footer{display:flex;justify-content:space-between;gap:6px;margin-top:7px}.pdg-footer button{border:0;background:none;color:#7598ba;font-size:9px;cursor:pointer}.pdg-footer button:hover{color:#d8ecff}
      .port-demo-launcher{position:fixed;z-index:12049;left:10px;top:92px;min-height:38px;padding:0 12px;border:1px solid rgba(34,211,238,.58);border-radius:10px;background:linear-gradient(135deg,#075985,#0f766e);box-shadow:0 12px 30px rgba(0,0,0,.34);color:white;font-size:11px;font-weight:900;cursor:pointer}
      .port-demo-target{outline:2px solid #22d3ee!important;outline-offset:4px!important;box-shadow:0 0 0 6px rgba(34,211,238,.12),0 0 32px rgba(34,211,238,.34)!important;animation:pdgTarget 1s ease-in-out infinite alternate!important}.port-demo-region{box-shadow:inset 4px 0 0 rgba(45,212,191,.88),0 0 0 1px rgba(45,212,191,.24)!important}.port-demo-result{outline:2px solid #34d399!important;outline-offset:4px!important;box-shadow:0 0 28px rgba(52,211,153,.26)!important}@keyframes pdgTarget{to{outline-color:#fbbf24;box-shadow:0 0 0 8px rgba(251,191,36,.08),0 0 38px rgba(34,211,238,.4)}}
      @media(max-width:1000px){body.port-demo-open,body.port-demo-collapsed{padding-left:0!important}.port-demo-guide{top:auto;left:8px;right:8px;bottom:8px;width:auto;height:min(52vh,520px)}.port-demo-guide[data-collapsed="true"]{left:8px;right:auto;width:50px;height:50px}.port-demo-launcher{left:8px;top:auto;bottom:8px}.pdg-steps{grid-template-columns:1fr 1fr}}@media(prefers-reduced-motion:reduce){.port-demo-target{animation:none!important}.pdg-progress i{transition:none}}
    `;
    document.head.appendChild(style);
  }

  function build(){
    root = document.createElement("aside");
    root.className = "port-demo-guide";
    root.dataset.demoGuide = "ready";
    root.dataset.version = VERSION;
    root.innerHTML = `<div class="pdg-head"><div class="pdg-wide"><div class="pdg-kicker">LIVE DEMO CONTROL</div><div class="pdg-title">港口数字孪生 · 演示引导</div></div><div class="pdg-head-actions"><button class="pdg-icon pdg-collapse" data-guide-action="collapse" title="折叠">‹</button><button class="pdg-icon pdg-close" data-guide-action="close" title="关闭">×</button></div></div><div class="pdg-wide pdg-body"><div class="pdg-progress-meta"><span data-guide-index></span><span>原生按钮联动</span></div><div class="pdg-progress"><i data-guide-progress></i></div><ol class="pdg-steps" data-guide-steps></ol><button type="button" data-demo-evidence-open>场景算法与数据清单</button><div class="pdg-phase" data-guide-phase></div><div class="pdg-step-title" data-guide-title></div><div class="pdg-location" data-guide-location></div><div class="pdg-speech" data-guide-speech></div><div class="pdg-wait" data-guide-wait></div><div class="pdg-status" data-guide-status></div><button class="pdg-primary" data-guide-action="primary"></button><div class="pdg-controls"><button data-guide-action="retry">重试本步</button><button data-guide-action="reset-step">重置本步</button><button data-guide-action="reset-all">重置全部</button><button class="pdg-fallback" data-guide-action="fallback">保底下一步</button></div><div class="pdg-footer"><button data-guide-action="restart">重新开始</button><button data-guide-action="stop-all">全部停止</button><button data-guide-action="finish">结束演示</button></div></div>`;
    launcher = document.createElement("button");
    launcher.className = "port-demo-launcher";
    launcher.dataset.demoGuideLauncher = "ready";
    launcher.textContent = "演示开始";
    launcher.addEventListener("click", startGuide);
    root.addEventListener("click", handleGuideClick);
    document.body.append(root, launcher);
  }

  function handleGuideClick(event){
    const stepButton = event.target.closest("[data-guide-step]");
    if(stepButton){ jumpToStep(Number(stepButton.dataset.guideStep), "引导目录"); return; }
    const button = event.target.closest("[data-guide-action]");
    if(!button) return;
    const action = button.dataset.guideAction;
    if(action === "primary") runCurrent();
    else if(action === "retry") retryCurrent();
    else if(action === "reset-step") resetCurrent();
    else if(action === "reset-all" || action === "restart") resetAll();
    else if(action === "fallback") advanceCurrent(true);
    else if(action === "stop-all") stopAllNativeActivity();
    else if(action === "finish") finishGuide();
    else if(action === "close"){ state.open = false; saveState(); render(); }
    else if(action === "collapse"){ state.collapsed = !state.collapsed; saveState(); render(); }
  }
  function startGuide(){ state = {...DEFAULT_STATE,open:true}; saveState(); render(); enterCurrentStep(); }
  function clearTimer(){ if(receiptTimer !== null) window.clearTimeout(receiptTimer); receiptTimer = null; waitingTrainingContract = false; }
  function clearHighlights(){
    document.querySelectorAll(".port-demo-target").forEach(node => node.classList.remove("port-demo-target"));
    document.querySelectorAll(".port-demo-region").forEach(node => node.classList.remove("port-demo-region"));
    document.querySelectorAll(".port-demo-result").forEach(node => node.classList.remove("port-demo-result"));
  }
  function navigateTo(route, executeAfterNavigation){
    state.pendingExecute = executeAfterNavigation ? currentStep()?.id || "" : "";
    saveState();
    const selector = route === "/" ? '#btnBackToHome,a[href="/"]' : `[data-route="${route}"],a[href="${route}"]`;
    const nav = findTarget(selector);
    if(nav){ programmaticClick = true; nav.click(); programmaticClick = false; }
    else window.location.assign(route);
  }
  function selectorForPage(step){
    if(step?.landingReceipt && !pathIs(step.route)) return step.sourceSelector || "";
    return step?.selector || "";
  }
  function positionStep(step, smooth=true){
    if(!step) return false;
    clearHighlights();
    const onDestination = pathIs(step.route);
    if(!onDestination && !step.landingReceipt) return false;
    const regionSelector = onDestination ? step.region : step.sourceRegion;
    const region = regionSelector ? document.querySelector(regionSelector) : null;
    const reveal = step.reveal ? findTarget(step.reveal) : null;
    if(reveal && !isVisible(region)){ programmaticClick = true; reveal.click(); programmaticClick = false; }
    const revealMenu = step.revealMenu ? findTarget(step.revealMenu) : null;
    if(revealMenu && !isVisible(findTarget(selectorForPage(step)))){ programmaticClick = true; revealMenu.click(); programmaticClick = false; }
    const target = findTarget(selectorForPage(step));
    const focus = target || region;
    if(focus) focus.scrollIntoView({block:"center",inline:"nearest",behavior:"auto"});
    region?.classList.add("port-demo-region");
    target?.classList.add("port-demo-target");
    return !!target;
  }
  function enterCurrentStep(){
    const step = currentStep();
    if(!step) return;
    if(step.landingReceipt){
      if(pathIs(step.route)){ scheduleLandingReceipt(step); return; }
      if(findTarget(step.sourceSelector)){ window.setTimeout(() => positionStep(step, true), 80); return; }
      if(!pathIs("/")){ navigateTo("/", false); return; }
      state.statusText = "正在等待原生模块菜单加载；加载后会自动定位当前入口。";
      saveState(); render();
      return;
    }
    if(!pathIs(step.route)){ navigateTo(step.route, false); return; }
    window.setTimeout(() => { prepareStep(step); positionStep(step, true); }, 80);
  }
  function scheduleLandingReceipt(step){
    window.setTimeout(() => {
      if(currentStep()?.id !== step.id || !pathIs(step.route)) return;
      const region = document.querySelector(step.region);
      if(!region || !isVisible(region)) return;
      const result = document.querySelector(step.result || step.region);
      const target = result && isVisible(result) ? step.result : step.region;
      completeCurrent({stepId:step.id,target,message:`已进入${step.location}；跨页进度、台词和高亮已同步。`});
    }, 260);
  }
  function prepareStep(step){
    if(step?.id !== "xiaoyi-command") return;
    const widget = document.querySelector(".xiaoyi-sprite-root");
    const input = widget?.querySelector(".xiaoyi-sprite-input");
    widget?.classList.add("open");
    if(input){
      input.value = "小鸥，请启动演示级快速训练，平衡成本与碳排，保留严格安全边界和人工确认";
      input.dispatchEvent(new Event("input",{bubbles:true}));
      input.focus({preventScroll:true});
    }
  }
  function jumpToStep(index, origin){
    if(!Number.isInteger(index) || index < 0 || index >= STEPS.length) return;
    clearTimer(); actionBusy = false; state.open = true; state.collapsed = false;
    state.current = index; state.phase = state.completed.includes(STEPS[index].id) ? "done" : "ready";
    state.statusText = `${origin}已切换到第 ${index + 1} 步；台词、按钮和目标区域已同步。`;
    state.pendingExecute = ""; saveState(); render(); enterCurrentStep();
  }
  function startReceiptWait(step, origin){
    clearTimer();
    state.phase = "waiting";
    state.statusText = origin === "page" ? `已识别页面原按钮：${step.label}。正在等待真实结果…` : `已点击真实按钮：${step.label}。当前台词保持不变，正在等待结果…`;
    saveState(); render();
    receiptTimer = window.setTimeout(() => {
      if(currentStep()?.id !== step.id || state.phase !== "waiting") return;
      state.phase = "error";
      state.statusText = "等待结果超时，流程已自动解锁。可以重试、重置本步或保底进入下一步。";
      saveState(); render();
    }, step.timeoutMs || 12000);
  }
  function runCurrent(){
    const step = currentStep();
    if(!step){ resetAll(); return; }
    if(state.phase === "done"){ advanceCurrent(false, true); return; }
    if(state.phase === "waiting" || actionBusy) return;
    if(step.landingReceipt){
      if(pathIs(step.route)){ scheduleLandingReceipt(step); return; }
      const sourceTarget = findTarget(step.sourceSelector);
      if(!sourceTarget || sourceTarget.disabled){
        if(!pathIs("/")){
          state.statusText = `正在返回主界面定位“${step.label}”原生入口。`;
          state.phase = "ready"; saveState(); render(); navigateTo("/", true); return;
        }
        state.phase = "error";
        state.statusText = "原生模块菜单尚未就绪。页面没有锁死，可重试本步或保底继续。";
        saveState(); render(); return;
      }
      actionBusy = true; startReceiptWait(step,"guide"); positionStep(step,true);
      window.setTimeout(() => {
        programmaticClick = true; sourceTarget.click(); programmaticClick = false;
        actionBusy = false;
      },180);
      return;
    }
    if(!pathIs(step.route)){
      state.statusText = `正在同一窗口进入${step.location}，到达后自动执行本步。`;
      state.phase = "ready"; saveState(); render(); navigateTo(step.route, true); return;
    }
    if(step.id === "training-review" && document.querySelector("#selDataset")?.getAttribute("data-training-ready") === "false"){
      clearTimer(); waitingTrainingContract = true; state.phase = "waiting";
      state.statusText = "正在校验训练数据集和档案参数；就绪后自动打开人工确认门，不会启动训练。";
      saveState(); render();
      receiptTimer = window.setTimeout(() => {
        if(!waitingTrainingContract || currentStep()?.id !== step.id) return;
        clearTimer(); state.phase = "error";
        state.statusText = "训练数据契约尚未就绪；可刷新接入口后重试，未打开确认门或启动训练。";
        saveState(); render();
      }, 60000);
      return;
    }
    const target = findTarget(step.selector);
    if(!target || target.disabled){
      state.phase = "error";
      state.statusText = "当前真实按钮尚未就绪。页面没有锁死，可重试本步或使用保底下一步。";
      saveState(); render(); positionStep(step, false); return;
    }
    actionBusy = true;
    startReceiptWait(step, "guide");
    positionStep(step, true);
    window.setTimeout(() => {
      programmaticClick = true; target.click(); programmaticClick = false;
      actionBusy = false; render();
    }, 180);
  }
  function completeCurrent(detail={}){
    const step = currentStep();
    if(!step || detail.stepId !== step.id) return;
    if(step.id === "twin3d-showcase" && document.querySelector('#sceneHost')?.dataset.renderState !== 'ready') return;
    clearTimer(); actionBusy = false;
    if(!state.completed.includes(step.id)) state.completed.push(step.id);
    state.skipped = state.skipped.filter(id => id !== step.id);
    state.phase = "done";
    state.statusText = detail.message || "页面已经完成本步变化；请保持当前台词，讲完后再进入下一步。";
    saveState(); clearHighlights();
    const result = document.querySelector(detail.target || step.result || step.region);
    result?.classList.add("port-demo-result");
    result?.scrollIntoView({block:"center",inline:"nearest",behavior:"auto"});
    render();
  }
  function advanceCurrent(fallback, execute=false){
    window.dispatchEvent(new Event("demo-evidence-close"));
    const step = currentStep();
    if(!step || (!fallback && state.phase !== "done")) return;
    clearTimer();
    if(fallback && !state.skipped.includes(step.id)) state.skipped.push(step.id);
    state.current += 1;
    state.phase = state.current >= STEPS.length ? "finished" : "ready";
    state.statusText = state.current >= STEPS.length ? (STEPS.every(item => state.completed.includes(item.id)) ? "数字孪生核心闭环与能力速览已完成。所有生产权限仍保持关闭。" : "演示已结束，仍有未完成步骤；请按清单补验。生产权限仍保持关闭。") : fallback ? "已记录保底推进；上一动作未标记为成功，当前台词和按钮已同步到下一步。" : "已进入下一步，页面正在定位对应业务区域。";
    state.pendingExecute = ""; saveState(); render();
    if(state.current < STEPS.length) {
      if(execute) {
        if(!pathIs(currentStep().route) && !currentStep().landingReceipt){ navigateTo(currentStep().route,true); return; }
        prepareStep(currentStep()); positionStep(currentStep(),true);
        window.setTimeout(runCurrent,220);
      } else enterCurrentStep();
    } else clearHighlights();
  }
  function retryCurrent(){
    const step = currentStep(); if(!step) return;
    clearTimer(); state.phase = "ready"; state.statusText = "本步已解锁，正在重新触发同一个真实页面按钮。";
    saveState(); render(); runCurrent();
  }
  function resetCurrent(){
    const step = currentStep(); if(!step) return;
    clearTimer();
    if(step.id === "situation"){
      const play = document.querySelector("#story-play");
      if(play && /播放中/.test(play.textContent || "")){ programmaticClick = true; play.click(); programmaticClick = false; }
    }
    if(["training-start","training-status"].includes(step.id) && pathIs("/rl-panel")){
      const reset = document.querySelector("#btnResetTrain");
      if(reset && !reset.disabled){ programmaticClick = true; reset.click(); programmaticClick = false; }
    }
    state.completed = state.completed.filter(id => id !== step.id);
    state.skipped = state.skipped.filter(id => id !== step.id);
    state.phase = "ready"; state.statusText = "本步已重置，可以重新执行真实页面按钮。";
    saveState(); render(); enterCurrentStep();
  }
  function resetAll(){
    clearTimer(); stopNativeActivity(false); state = {...DEFAULT_STATE,open:true}; saveState(); render();
    if(pathIs("/")){
      const home = findTarget('.nav-trigger[data-direct-target="#home-hero"]');
      if(home){ programmaticClick = true; home.click(); programmaticClick = false; }
      else window.PortModuleNavigation?.open?.("home-hero",{history:"replace"});
      window.setTimeout(enterCurrentStep,80);
    }else navigateTo("/", false);
  }
  function stopNativeActivity(updateStatus=true){
    const play = document.querySelector("#story-play");
    if(play && /播放中/.test(play.textContent || "")){ programmaticClick = true; play.click(); programmaticClick = false; }
    const reset = document.querySelector("#btnResetTrain");
    if(reset && !reset.disabled){ programmaticClick = true; reset.click(); programmaticClick = false; }
    if(updateStatus){
      state.phase = currentStep() ? "ready" : "finished";
      state.statusText = reset ? "已调用 RL 面板原生“重置”按钮并停止故事回放；取消结果会由后端日志确认。" : "已停止当前页面的故事回放；本页没有由引导启动的服务或训练任务。";
      saveState(); render();
    }
  }
  function finishGuide(){
    clearTimer();
    stopNativeActivity(false);
    clearHighlights(); state.open = false; state.collapsed = false; state.pendingExecute = ""; saveState(); render();
  }
  function render(){
    if(!root) return;
    const step = currentStep();
    const finished = !step;
    const completedCount = STEPS.filter(item => state.completed.includes(item.id)).length;
    const allCompleted = completedCount === STEPS.length;
    root.hidden = !state.open; launcher.hidden = state.open; root.dataset.collapsed = String(!!state.collapsed);
    document.body.classList.toggle("port-demo-open",state.open && !state.collapsed);
    document.body.classList.toggle("port-demo-collapsed",state.open && state.collapsed);
    root.querySelector("[data-guide-index]").textContent = finished ? `${completedCount} / ${STEPS.length} 已完成` : `${state.current + 1} / ${STEPS.length}`;
    root.querySelector("[data-guide-progress]").style.width = `${Math.round(((finished ? completedCount : state.current + 1) / STEPS.length) * 100)}%`;
    root.querySelector("[data-guide-steps]").innerHTML = STEPS.map((item,index) => {
      const done = state.completed.includes(item.id), skipped = state.skipped.includes(item.id), active = index === state.current;
      const className = ["pdg-step",active?"active":"",done?"done":"",skipped?"skipped":""].filter(Boolean).join(" ");
      const mark = done ? "✓" : skipped ? "!" : index + 1;
      const stepStatus = active ? (state.phase === "done" ? "已执行" : state.phase === "waiting" ? "执行中" : "当前") : done ? "完成" : skipped ? "跳过" : "";
      const divider = index === STEPS.findIndex(candidate => candidate.group === "quick") ? '<li class="pdg-section-divider">以下为能力速览 · 完成情况以回执为准</li>' : '';
      return `${divider}<li class="${className}" data-guide-step="${index}" title="跳到此步骤"><i>${mark}</i><b>${escapeHtml(item.title)}</b><em>${stepStatus}</em></li>`;
    }).join("");
    root.querySelector("[data-guide-phase]").textContent = finished ? (allCompleted ? "演示完成" : "演示待补验") : step.phase;
    root.querySelector("[data-guide-title]").textContent = finished ? (allCompleted ? "港口数字孪生演示闭环已完成" : "演示已结束，尚有未完成步骤") : step.title;
    root.querySelector("[data-guide-location]").textContent = finished ? "公开数据 · 离线训练 · 校准模拟" : step.location;
    root.querySelector("[data-guide-speech]").textContent = finished && !allCompleted ? `实际完成 ${completedCount}/${STEPS.length} 步；跳过或未执行的步骤不计为通过，可点击清单继续补验。` : finished ? "本次演示先完成态势、官方数据、RL 数据集与契约、舷鸥下发、人工确认、真实训练回执、历史收敛、业务指标、安全门和审计回溯，再快速展示 3D 孪生、多港区、ESG、策略编排、运营副驾、V3、项目联动、未来舱和开放接口；所有结论保留适用边界，生产权限没有被开启。" : step.speech;
    root.querySelector("[data-guide-wait]").textContent = finished ? "可以重新开始，或关闭引导继续查看页面。" : `可见变化：${step.wait}`;
    const status = root.querySelector("[data-guide-status]"); status.dataset.state = state.phase; status.textContent = state.statusText;
    const primary = root.querySelector('[data-guide-action="primary"]');
    primary.classList.toggle("is-next",state.phase === "done" || finished); primary.disabled = state.phase === "waiting" || actionBusy;
    primary.textContent = finished ? "↻ 重新开始演示" : state.phase === "done" ? (state.current === STEPS.length - 1 ? "✓ 讲解完成，结束演示" : "✓ 下一步并演示") : state.phase === "waiting" ? "正在执行并等待回执…" : state.phase === "error" ? `↻ 重试：${step.label}` : `▶ 点击：${step.label}`;
  }
  function onPageClick(event){
    // Both guide-triggered and direct page clicks flow through the same step
    // mapping. Re-applying the same waiting state is idempotent and avoids a
    // race where a fast user click lands while a menu is being positioned.
    if(!state.open) return;
    const target = event.target;
    if(!(target instanceof Element)) return;
    const triggerIndex = STEPS.findIndex(step => pathIs(step.route) && step.triggerSelector && target.closest(step.triggerSelector));
    if(triggerIndex >= 0){
      if(programmaticClick) return;
      const triggerStep = STEPS[triggerIndex];
      clearTimer(); actionBusy = false; state.current = triggerIndex; state.phase = state.completed.includes(triggerStep.id) ? "done" : "ready"; state.open = true;
      state.statusText = `已识别页面菜单：${triggerStep.location}。请选择高亮功能，或点击引导操作按钮。`;
      state.pendingExecute = ""; saveState(); render(); window.setTimeout(() => positionStep(triggerStep,false),0); return;
    }
    const index = STEPS.findIndex(step => step.landingReceipt ? target.closest(step.sourceSelector) : pathIs(step.route) && target.closest(step.selector));
    if(index < 0) return;
    const step = STEPS[index];
    if(step.id === "xiaoyi-command"){
      const command = document.querySelector(".xiaoyi-sprite-input")?.value || "";
      if(!/(训练|强化学习|\bRL\b)/i.test(command)) return;
    }
    clearTimer(); actionBusy = false; state.current = index; state.phase = "ready"; state.open = true;
    state.statusText = `已从页面原按钮同步到第 ${index + 1} 步：${step.title}。`;
    state.pendingExecute = ""; saveState(); render(); positionStep(step, false); startReceiptWait(step,"page");
  }

  window.portDtDemoReceipt = function(stepId,detail={}){
    window.dispatchEvent(new CustomEvent(RECEIPT_EVENT,{detail:{...detail,stepId}}));
  };
  window.portDtDemoGuideIsOpen = () => state.open === true;
  window.addEventListener(RECEIPT_EVENT,event => completeCurrent(event.detail || {}));
  window.addEventListener("port-dt:training-contract-ready",() => {
    if(!state.open || !waitingTrainingContract || currentStep()?.id !== "training-review" || state.phase !== "waiting") return;
    clearTimer(); state.phase = "ready"; runCurrent();
  });
  window.addEventListener("port-dt:module-menu-ready",() => { if(state.open && currentStep()) enterCurrentStep(); });
  document.addEventListener("click",onPageClick,true);

  function install(){
    if(!document.body) return;
    addStyle(); build(); render();
    if(state.open && currentStep()) window.setTimeout(enterCurrentStep,80);
    if(state.pendingExecute && currentStep()?.id === state.pendingExecute){
      state.pendingExecute = ""; state.phase = "ready"; saveState(); window.setTimeout(runCurrent,220);
    }
  }
  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded",install,{once:true});
  else install();
})();
