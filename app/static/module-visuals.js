/* Read-only visual workspaces. Routing, original controls and audit logic remain their owners. */
(function(){
  'use strict';
  if(window.__portModuleVisuals)return;window.__portModuleVisuals=true;
  const configs={
    'exec-cockpit':{title:'经营态势驾驶舱',en:'EXECUTIVE / OPERATING INTELLIGENCE',sub:'经营、作业与能源放在同一时间轴，先看趋势，再核验收益口径。',kind:'executive',chart:'单码头场景窗口 · 负荷与吞吐',side:'展示源 · 作业快照'},
    'platform-map':{title:'从港口现场到运营决策',en:'PLATFORM / CONNECTED ARCHITECTURE',sub:'沿着设备、数据、模型、应用逐层看清系统关系。',kind:'architecture',chart:'四层能力拓扑',side:'平台能力边界'},
    'dev-ecosystem':{title:'开放能力与应用生态',en:'DEVELOPER / CAPABILITY EXPLORER',sub:'按业务域查看已注册接口，连接应用与数据服务。',kind:'apps',chart:'接口能力分布',side:'运行时注册表'},
    'multiport-section':{title:'跨港区迁移与复用',en:'MULTI-PORT / TRANSFER READINESS',sub:'每个节点是一份端口画像，点选查看环境版本与现场接入要求。',kind:'profiles',chart:'画像关系网络 · 非地理位置',side:'端口画像详情'},
    'esg-section':{title:'碳排来源，看得清楚',en:'ESG / CARBON INTELLIGENCE',sub:'公开锚点、工程电量、排放因子与碳强度，沿来源链逐项核验。',kind:'carbon',chart:'电量 × 排放因子 → 碳排',side:'碳证据摘要'},
    'cmp-section':{title:'从排放来源到审计报表',en:'COMPLIANCE / REPORTING WORKSPACE',sub:'先核验计量边界，再进入原有月度、季度与分项报表。',kind:'compliance',chart:'报表证据链',side:'审计数据就绪态'},
    'twin3d-section':{title:'三维孪生与能碳推演',en:'DIGITAL TWIN / SCENARIO EXPLORER',sub:'观察设施结构，再联动下方现在、预测与策略仿真的原有曲线。',kind:'spatial',scene:'port',chart:'作业空间 · 港区设施示意',side:'当前校准回放'},
    'story-section':{title:'一次决策的前因后果',en:'STORY / PAIRED BLIND-TEST REPLAY',sub:'先看同窗盲测的整体差异，再用下方时间轴逐小时回放。',kind:'advantage',chart:'相对 FCFS 改善率与 95% 区间',side:'回放与比较条件'},
    'kpi-section':{title:'港口能源与碳排全貌',en:'ENERGY / LOAD INTELLIGENCE',sub:'负荷趋势与设备构成互相补充，工程阈值仍由下方原有控件设置。',kind:'energy',chart:'标定窗口 · 负荷与吞吐',side:'本次回放快照'},
    'realtime-section':{title:'从数据脉搏到运行判断',en:'TELEMETRY / CONTINUOUS REPLAY',sub:'设备负荷、时间口径和模型建议同屏查看。',kind:'energy',chart:'标定窗口 · 负荷与吞吐',side:'快照与来源时间'},
    'twin-section':{title:'策略进入港口作业空间',en:'TWIN / MODEL TO OPERATIONS',sub:'把模型建议与设施结构连接，继续在下方执行原有推演。',kind:'spatial',scene:'port',chart:'作业空间 · 示意布局',side:'策略回放快照'},
    'strategy-exec-module':{title:'从策略依据到执行回执',en:'STRATEGY / DECISION PIPELINE',sub:'看清每一步需要的证据，再使用下方原有策略与执行控件。',kind:'pipeline',chart:'策略执行链 · 能力流程示意',side:'当前模块状态',steps:['态势与预测','策略与反事实','工程安全门','审批与回执'],metrics:[['当前场景','strategy-scene-chip'],['策略来源','strategy-source-chip'],['审批工单','qc-pending-count'],['生产权限','strategy-ready-head']]},
    'rlops-section':{title:'模型运行与上线准入',en:'RL OPS / POLICY LIFECYCLE',sub:'训练、盲测、影子运行与现场准入分层呈现。',kind:'advantage',chart:'所选 SAC · 三种子盲测改善率',side:'模型与数据协议'},
    'exec-wrap':{title:'执行链路与人工确认',en:'EXECUTION / HUMAN IN THE LOOP',sub:'每个动作都需要合法设备、工程约束、确认与回执。',kind:'pipeline',chart:'设备命令生命周期 · 流程示意',side:'执行证据',steps:['设备白名单','约束与暂存','异人确认','回执与回滚'],metrics:[['审批台账','qc-pending-count'],['最近工单','qc-last-job'],['设备控制',null,'待现场接入'],['执行权限',null,'未授予']]},
    'mas-section':{title:'多智能体协同工作台',en:'MULTI-AGENT / COORDINATION',sub:'沿资源、目标、协调、安全投影查看协同关系。',kind:'pipeline',chart:'资源与决策协调 · 流程示意',side:'协同场景证据',steps:['资源与作业状态','多目标建议','冲突协调','安全投影'],metrics:[['吞吐量','mas-kpi-teu'],['等待 P95','mas-kpi-wp95'],['功率','mas-kpi-kw'],['能耗','mas-kpi-kwh']]},
    'twinlab-section':{title:'场景工厂与压力试验',en:'TWINLAB / SCENARIO COVERAGE',sub:'按场景查看软件覆盖与待接合同，再进入原有演练与协同试验。',kind:'coverage',chart:'软件场景覆盖矩阵',side:'覆盖范围'},
    'train-section':{title:'训练证据与数据隔离',en:'TRAINING / EXPERIMENT OBSERVATORY',sub:'数据切分、算法记录和盲测边界与原始训练曲线相互对应。',kind:'training',chart:'时间顺序数据切分 · 不随机打乱',side:'训练证据概览'},
    'deep-section':{title:'港口业务能力全景',en:'OPERATIONS / DOMAIN COVERAGE',sub:'按业务域查看模型输出和现场替换需求。',kind:'domains',chart:'业务域能力矩阵',side:'业务能力统计'},
    'yl-section':{title:'堆场照明 · 空间与策略',en:'YARD LIGHTING / ASSET TWIN',sub:'高杆灯、堆位与照度约束对应，查看模型请求和安全投影后的动作。',kind:'facility',scene:'lighting',endpoint:'yard-lighting',chart:'堆场照明三维剖面 · 布局示意',side:'模型设定与约束'},
    'hvac-section':{title:'冷站运行 · 从设备到设定值',en:'HVAC / PLANT TWIN',sub:'冷机、冷却塔与供回水结构联动呈现，模型设定保持来源可查。',kind:'facility',scene:'hvac',endpoint:'hvac',chart:'冷站设备三维剖面 · 结构示意',side:'模型设定与约束'},
    'sbess-section':{title:'岸电储能 · 船岸能源连接',en:'SHORE POWER / ENERGY TWIN',sub:'从电池舱到 PCS、并网点与船舶岸电，沿连接关系核验策略。',kind:'facility',scene:'shore',endpoint:'shore-bess',chart:'船岸能源三维剖面 · 结构示意',side:'模型设定与约束'},
    'be-section':{title:'储能调度 · 电池与电网之间',en:'BESS / DISPATCH TWIN',sub:'电池舱、变流器和并网点同屏观察，SoC 仅取模型返回值。',kind:'facility',scene:'bess',endpoint:'bess-energy',chart:'储能系统三维剖面 · 结构示意',side:'模型设定与约束'},
    'yc-section':{title:'场桥作业 · 空间与能效',en:'YARD CRANE / OPERATIONS TWIN',sub:'轨道、门腿、起升小车和堆位对应，作业保留与能耗改善分别核验。',kind:'facility',scene:'crane',endpoint:'yard-crane',chart:'场桥作业三维剖面 · 机构示意',side:'模型设定与约束'},
    'ai-trust-scenes':{title:'让策略可信，有据可查',en:'AI TRUST / EVIDENCE GATES',sub:'把离线盲测优势与现场授权分开核验。',kind:'advantage',chart:'相对 FCFS 改善率与 95% 区间',side:'证据协议与准入'},
    'monitoring-section':{title:'数据质量与分布漂移',en:'MONITORING / QUALITY OBSERVATORY',sub:'比较基线与近期窗口的分布，配合原有异常扫描定位问题。',kind:'monitoring',chart:'PSI · 基线与近期分布',side:'当前分析窗口'},
    'opsx-section':{title:'运行治理 · 由门禁到回退',en:'OPSX / OPERATING GUARDRAILS',sub:'将运行、质量、审批与审计串成可检查的链路。',kind:'pipeline',chart:'策略运行治理 · 流程示意',side:'运行就绪态',steps:['质量与漂移','候选与门禁','影子与灰度','审计与回退'],metrics:[['运行阶段','opsx-v3-phase'],['流量比例','opsx-v3-traffic'],['准入门禁','opsx-v3-gates'],['现场状态','opsx-v3-site']]},
    'ext-section':{title:'港口外部环境与业务信号',en:'EXTERNAL / ENVIRONMENTAL INTELLIGENCE',sub:'风速、潮位和浪高按来源时间展示，公开再分析与工程派生分别标注。',kind:'environment',chart:'公开窗口 · 风速 / 潮位 / 浪高',side:'来源与可用性'},
    'mlops-section':{title:'从数据到模型的可追溯链',en:'MLOPS / MODEL LINEAGE',sub:'数据切分、权重、盲测和部署门禁建立可核验关系。',kind:'training',chart:'时间顺序切分 · 模型血缘入口',side:'模型与产物'},
    'g-section':{title:'治理控制与风险全景',en:'GOVERNANCE / CONTROL MATRIX',sub:'状态直接读取原有治理矩阵，点开下方证据继续核验。',kind:'governance',chart:'治理控制矩阵',side:'治理与审计摘要'},
    'standalone-rl':{title:'算法、数据与独立盲测',en:'RL LAB / EXPERIMENT OVERVIEW',sub:'先了解正式证据与数据协议，再在下方创建训练或评测任务。',kind:'training',chart:'公开目标域 · 时间顺序数据隔离',side:'算法与训练证据'},
    'standalone-v3':{title:'业务域与模型证据全景',en:'DECISION CENTER / DOMAIN MAP',sub:'从业务域进入模型能力与现场替换要求。',kind:'domains',chart:'已实现业务域 · 证据矩阵',side:'模型输出与现场边界'},
    'standalone-future':{title:'让推演回到真实港口结构',en:'FUTURE DECK / SPATIAL CONTEXT',sub:'先观察作业与能源空间，再通过原有“启动推演”生成候选与回执。',kind:'spatial',scene:'port',chart:'港区作业结构 · 示意布局',side:'校准回放快照'},
    'standalone-hub':{title:'跨系统服务与证据流',en:'LINKAGE / SERVICE TOPOLOGY',sub:'服务在线状态来自本页原有探测，任务仍由原有按钮触发。',kind:'services',chart:'系统服务拓扑 · 逻辑关系',side:'当前连接状态'},
    'standalone-copilot':{title:'运营答案背后的证据链',en:'COPILOT / CONTEXT TO ANSWER',sub:'来源、模型、门禁和上下文哈希逐层核验，再进入下方任务。',kind:'copilot',chart:'从上下文到可交接答案',side:'本次上下文状态'}
  };
  const sourceName=d=>d.mode==='public_parameter_calibrated_demo'?'公开参数标定演示':'外部展示快照';
  const colors=['#73d9c4','#7aafe1','#e1b779','#a49bdd','#7eb2ad'];
  // Keep accepted model pages; strategy overview owns a separate task workspace.
  const preservedModelPages=new Set(['train-section','yl-section','hvac-section','sbess-section','be-section','yc-section','rlops-section','strategy-exec-module']);
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const valid=n=>typeof n==='number'&&Number.isFinite(n);
  const scaled=(n,f)=>valid(n)?n*f:null;
  const fmt=(n,d=1)=>valid(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
  const read=id=>id?document.getElementById(id)?.textContent.trim()||'等待读取':'—';
  const cache=new Map();let active=null;
  async function json(url,refresh=false){
    const old=cache.get(url);if(!refresh&&old&&Date.now()-old.time<60000)return old.promise;
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),45000);
    const promise=fetch(url,{cache:'no-store',signal:controller.signal}).then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();}).catch(e=>{cache.delete(url);throw e;}).finally(()=>clearTimeout(timer));
    cache.set(url,{time:Date.now(),promise});return promise;
  }
  function stats(host,rows){host.innerHTML=`<div class="mv-metrics">${rows.map(([name,value,unit])=>`<div class="mv-stat"><small>${esc(name)}</small><strong${String(value).length>10?' style="font-size:14px;line-height:1.6"':''}>${esc(value)}</strong><em>${esc(unit||'')}</em></div>`).join('')}</div>`;}
  function note(host,text){host.insertAdjacentHTML('beforeend',`<p class="mv-side-note">${esc(text)}</p>`);}
  function flow(host,nodes){host.innerHTML=`<div class="mv-flow">${nodes.map(([title,sub],i)=>`<div class="mv-flow-node"><i>0${i+1} / LINK</i><b>${esc(title)}</b><span>${esc(sub)}</span></div>`).join('')}</div>`;}
  function matrix(host,rows){host.innerHTML=`<div class="mv-matrix">${rows.map(([name,status,pass])=>`<div class="mv-cell${pass?' pass':''}"><b>${esc(name)}</b><span>${esc(status)}</span></div>`).join('')}</div>`;}
  function chart(ctx,host,option,alt){
    if(!window.echarts){host.innerHTML=`<div class="mv-empty"><b>图表引擎未加载</b><span>可以继续查看下方原始数据与证据。</span></div>`;return;}
    host.setAttribute('role','img');host.setAttribute('aria-label',alt);const plot=window.echarts.init(host,null,{renderer:'canvas'});ctx.charts.push(plot);
    plot.setOption({animation:false,color:colors,textStyle:{color:'#8aa9b9',fontFamily:'system-ui'},tooltip:{trigger:'axis',backgroundColor:'#122d3b',borderColor:'#416473',textStyle:{color:'#d7eaf0',fontSize:11}},grid:{left:64,right:32,top:38,bottom:40},...option});
    const observer=new ResizeObserver(()=>{if(host.getClientRects().length)plot.resize();});observer.observe(host);ctx.chartClean.push(()=>observer.disconnect());return plot;
  }
  const axis={axisLine:{lineStyle:{color:'#2c4654'}},axisTick:{show:false},splitLine:{lineStyle:{color:'#233d4a'}},axisLabel:{color:'#7e9eae',fontSize:10}};
  function trace(ctx,host,rows,series,title){
    chart(ctx,host,{legend:{top:0,textStyle:{color:'#8eacbd',fontSize:10},itemWidth:14,itemHeight:6},xAxis:{...axis,type:'category',data:rows.map(p=>String(p.source_timestamp||p.timestamp||'').replace('T',' ').slice(5,16)),boundaryGap:false},yAxis:series.map((s,i)=>({...axis,type:'value',name:s.unit,position:i?'right':'left',splitLine:{show:i===0,lineStyle:{color:'#233d4a'}},scale:true})),series:series.map((s,i)=>({name:s.name,type:'line',data:rows.map(p=>valid(p[s.key])?p[s.key]:null),yAxisIndex:Math.min(i,1),showSymbol:rows.length<3,smooth:false,lineStyle:{width:2},areaStyle:i===0?{opacity:.06}:undefined,connectNulls:false}))},title);
  }
  function split(ctx,host,data){const d=data.dataset||{},parts=[d.train_rows,d.validation_rows,d.test_rows],total=parts.reduce((a,b)=>a+(valid(b)?b:0),0);if(!total){host.innerHTML='<div class="mv-empty">数据切分尚未返回</div>';return;}
    chart(ctx,host,{grid:{left:30,right:30,top:60,bottom:75},tooltip:{trigger:'item',formatter:p=>`${esc(p.seriesName)}：${fmt(p.value,0)} 行`},legend:{bottom:8,textStyle:{color:'#a3bcc8'}},xAxis:{show:false,max:total},yAxis:{show:false,type:'category',data:['时间轴']},series:parts.map((n,i)=>({name:['训练集','验证集','独立盲测集'][i],type:'bar',stack:'split',data:[n],barWidth:70,label:{show:true,position:'inside',formatter:()=>`${['训练','验证','盲测'][i]}\n${fmt(n,0)} 行`,fontSize:12,color:'#09212c'}}))},`时间顺序切分，训练 ${parts[0]} 行，验证 ${parts[1]} 行，独立盲测 ${parts[2]} 行`);
  }
  function advantage(ctx,host,data){const source=data.advantage?.selected?.metrics_relative_to_fcfs||{};const names={throughput_teu:'吞吐量',delay_index_mean:'延误指标',energy_cost:'总电能成本',carbon_kg:'总碳排',peak_kw:'需量峰值'};const rows=Object.entries(names).filter(([k])=>valid(source[k]?.mean));
    if(!rows.length){host.innerHTML='<div class="mv-empty">尚未取得配对盲测指标</div>';return;}
    const range=rows.flatMap(([k])=>[source[k].ci_low*100,source[k].ci_high*100]);
    chart(ctx,host,{grid:{left:92,right:52,top:20,bottom:32},tooltip:{trigger:'item',formatter:p=>{const [k,n]=rows[p.dataIndex],v=source[k];return `${n}<br>平均相对改善 ${fmt(v.mean*100,2)}%<br>95% CI [${fmt(v.ci_low*100,2)}, ${fmt(v.ci_high*100,2)}]%<br>n=${v.n} · 负值表示退化`;}},xAxis:{...axis,type:'value',min:Math.min(0,...range)*1.25,max:Math.max(0,...range)*1.3,axisLabel:{formatter:'{value}%',fontSize:9}},yAxis:{...axis,type:'category',inverse:true,data:rows.map(r=>r[1]),splitLine:{show:false}},series:[{type:'bar',barWidth:15,data:rows.map(([k])=>({value:source[k].mean*100,itemStyle:{color:source[k].mean>=0?colors[0]:colors[2]}})),label:{show:true,position:'right',fontSize:10,color:'#b5ccd7',formatter:p=>fmt(p.value,2)+'%'}},{type:'custom',data:rows.map(([k],i)=>[source[k].ci_low*100,source[k].ci_high*100,i]),renderItem:(p,api)=>{const a=api.coord([api.value(0),api.value(2)]),b=api.coord([api.value(1),api.value(2)]);return{type:'group',children:[{type:'line',shape:{x1:a[0],y1:a[1],x2:b[0],y2:b[1]},style:{stroke:'#e3ecef',lineWidth:1.2}},{type:'line',shape:{x1:a[0],y1:a[1]-5,x2:a[0],y2:a[1]+5},style:{stroke:'#e3ecef'}},{type:'line',shape:{x1:b[0],y1:b[1]-5,x2:b[0],y2:b[1]+5},style:{stroke:'#e3ecef'}}]};},z:5}]},'三种子配对盲测改善率，白色线段为95%置信区间，负值表示退化');
  }
  async function spatial(ctx){if(ctx.sceneLoading)return;ctx.sceneLoading=true;const attempt=ctx.sceneAttempts||0;ctx.sceneAttempts=attempt+1;const host=ctx.main.querySelector('.mv-spatial');try{const {createSpatial}=await import('/static/module-spatial.js?v=20261023-3');if(active!==ctx)return;ctx.spatial=await createSpatial(host,ctx.config.scene,label=>{ctx.main.querySelector('.mv-object-label').textContent=label;},attempt);if(active!==ctx){ctx.spatial.dispose();return;}if(valid(ctx.soc))ctx.spatial.updateSoc(ctx.soc);
    }catch(error){if(active===ctx){try{const {createCompatibleScene}=await import('/static/scene-compatible.js?v=20261031-1');if(active!==ctx)return;host.replaceChildren();ctx.spatial=createCompatibleScene(host,ctx.config.scene,(_key,label)=>{ctx.main.querySelector('.mv-object-label').textContent=label;});ctx.main.querySelector('.mv-object-label').textContent='兼容结构示意 · 点选设备查看名称 · 原有三维场景在支持 WebGL 时启用';if(valid(ctx.soc))ctx.spatial.updateSoc(ctx.soc);const motion=ctx.main.querySelector('[data-mv-action="motion"]');if(motion){motion.disabled=true;motion.textContent='静态兼容视图';}}catch(fallbackError){host.innerHTML=`<div class="mv-empty"><b>场景暂不可用</b><span>可继续查看右侧数据与下方原有图表，或点击“刷新视图”重试。</span></div>`;}}}finally{ctx.sceneLoading=false;}}
  function sceneMarkup(ctx){ctx.main.style.padding='0';ctx.main.innerHTML=`<div class="mv-scene-wrap"><div class="mv-scene-label">ASSET TWIN / SCHEMATIC</div><div class="mv-scene-toolbar"><button type="button" data-mv-action="top" aria-pressed="false">俯视</button><button type="button" data-mv-action="motion" aria-pressed="false">暂停示意动画</button><button type="button" data-mv-action="reset">复位</button></div><div class="mv-spatial" tabindex="0" role="img" aria-label="三维设施示意，拖动旋转，滚轮缩放，方向键旋转，点击设备查看名称"></div><div class="mv-object-label">拖动旋转 · 滚轮缩放 · 点选设备 · 动画为机构示意</div></div>`;spatial(ctx);}
  function redrawDom(ctx){
    const c=ctx.config;if(c.kind==='pipeline')stats(ctx.side,c.metrics.map(([n,id,value])=>[n,value||read(id),'原模块状态']));
    if(ctx.id==='exec-wrap'&&ctx.capabilities){const d=ctx.capabilities;stats(ctx.side,[['设备网关',d.enabled?'已配置':'关闭','后端能力状态'],['白名单设备',fmt(d.whitelisted_asset_count,0),'台 · 已配置设备'],['确认规则',d.requester_confirmer_must_differ?'异人双审':'待核验','下发前必须校验'],['现场验收',d.site_execution_accepted?'通过':'待验收','联锁 / 回读 / 回滚']]);note(ctx.side,d.execution_acceptance_status||'请核验原有执行安全边界。');}
    if(c.kind==='services'){
      const items=[['数字孪生平台',read('stPlatform'),'本地平台'],['舷鸥 · GullMate',read('stXiaoyi'),'原有连接探测'],['RL 接口',read('stRl'),'原有连接探测'],['航行模拟器',read('stSailing'),'原有连接探测']];stats(ctx.side,items);ctx.main.innerHTML=`<div class="mv-service-network"><div class="mv-service-hub">联动中枢<span>任务路由 · 证据回执</span></div><div class="mv-service-grid">${items.map(([n,v])=>`<div class="mv-service-node"><i></i><b>${esc(n)}</b><span>${esc(v)}</span></div>`).join('')}</div></div>`;note(ctx.side,'连接可用不等于任务完成；训练、模拟与回执仍需在下方操作。');
    }
    if(c.kind==='copilot'){
      const rows=[['数据来源',read('sourceRail'),'来源状态'],['模型策略',read('policyRail'),'模型状态'],['准入门禁',read('gateRail'),'监控状态'],['上下文哈希',read('hashRail').slice(0,16),'SHA-256 摘要']];stats(ctx.side,rows);flow(ctx.main,[['上下文事实',read('sourceRail')],['模型与门禁',read('gateRail')],['舷鸥推理',read('llmChip')],['交接与回执','结果在原任务工作区核验']]);
    }
    if(c.kind==='governance'){
      const rows=Array.from(document.querySelectorAll('#gov-control-table tr')).filter(tr=>tr.querySelector('td')).map(tr=>Array.from(tr.cells).map(td=>td.textContent.trim())).filter(r=>r.length>=3);
      if(rows.length)matrix(ctx.main,rows.map(r=>[r[1],r[2],/^(PASS|通过|passed|verified)$/i.test(r[2])]));else ctx.main.innerHTML='<div class="mv-empty"><b>治理控制读取中</b><span>原有治理矩阵返回后自动更新。</span></div>';
      stats(ctx.side,[['治理控制',read('gov-control-count'),'原始控制矩阵'],['已通过',read('gov-pass'),'离线证据'],['待现场',read('gov-pending'),'现场验收待补'],['下发决策',read('gov-decision'),'原始准入决策']]);
    }
  }
  async function load(ctx,refresh=false){
    ctx.loadStarted=performance.now();const c=ctx.config;ctx.status.textContent='读取当前模块数据…';ctx.refresh.disabled=true;
    try{
      if(['pipeline','services','copilot','governance'].includes(c.kind)){
        if(c.kind==='pipeline'){const descriptions=ctx.id==='exec-wrap'?['仅接收已登记设备','校验限值、时效与互锁','请求人与确认人不同','核验回读、回滚与审计']:['数据与时间口径','模型输出与反事实','校验后才可进入下一步','需独立现场确认与审计'];flow(ctx.main,c.steps.map((v,i)=>[v,descriptions[i]]));}
        if(ctx.id==='exec-wrap'){ctx.capabilities=await json('/api/actuators/capabilities',refresh);if(active!==ctx)return;}
        redrawDom(ctx);ctx.status.textContent='与原模块状态同步';return;
      }
      if(c.kind==='facility'){
        const d=await json(`/api/v3/modules/${c.endpoint}/evidence`,refresh);if(active!==ctx)return;
        const inf=d.current_model_output?.model_inference||{},st=inf.state||{},safe=inf.safety||{},contract=d.formal_training?.contract||{};
        const soc=valid(safe.soc)?safe.soc:valid(st.soc)?st.soc:null;ctx.soc=soc;if(valid(soc))ctx.spatial?.updateSoc(soc);
        const actions=inf.final_action||{},requested=inf.requested_action||{};
        const names={chws_C:'冷冻水 °C',sat_C:'送风 °C',chws_c:'冷冻水 °C',sat_c:'送风 °C',sp_Pa:'静压 Pa',CHWS_set:'冷冻水 °C',SAT_set:'送风 °C',SP_set:'静压 Pa',chws_sp_c:'冷冻水 °C',sat_sp_c:'送风 °C',static_pressure_pa:'静压 Pa',bess_kw:'储能功率 kW',reserve_kw:'备用功率 kW',flex_kw:'柔性负荷 kW',base_dimming_residual_ratio:'基础调光残差',activity_gain_ratio:'作业增益',weather_gain_ratio:'气象增益',power_cap_residual_pct:'功率上限残差',idle_timeout_residual_min:'待机超时残差 min'};
        let rows;
        if(c.scene==='hvac'){rows=[['冷冻水设定',fmt(d.replay_kpis?.model_chws_C,2),'°C · 模型建议'],['送风设定',fmt(d.replay_kpis?.model_sat_C,2),'°C · 模型建议'],['静压设定',fmt(d.replay_kpis?.model_sp_Pa,0),'Pa · 模型建议'],['安全约束',fmt(contract.hard_constraints?.length,0),'工程约束条数']];}
        else if(c.scene==='lighting')rows=[['最低照度',fmt(st.lux_min,1),'lux · 工程回放'],['平均调光',fmt(scaled(st.dimming_mean_ratio,100),1),'% · 工程回放'],['照明分区',fmt(st.zone_count,0),'模型状态 · 非沙盘数量'],['灯群功率',fmt(st.fleet_power_kw,1),'kW · 工程回放']];
        else if(c.scene==='crane')rows=[['当前作业',fmt(st.moves,0),'moves / 工程时间片'],['积压队列',fmt(st.backlog,0),'模型状态'],['作业设备',fmt(st.active_cranes,0),'台 · 工程回放'],['投影后上限残差',fmt(scaled(actions.power_cap_residual_pct,100),2),'% · 安全投影后']];
        else rows=[['电池 SoC',valid(soc)?fmt(soc*100,1):'—','% · 模型状态'],['储能功率',fmt(actions.bess_kw,1),'kW · 模型建议'],['备用功率',fmt(actions.reserve_kw,1),'kW · 缺失保留为空'],['电池温度',fmt(safe.temperature_c,1),'°C · 模型状态']];
        stats(ctx.side,rows);
        const actionEntries=Object.entries(actions).filter(([,v])=>valid(v));
        const residualKeys={chws_c:'chws_residual_c',sat_c:'sat_residual_c',static_pressure_pa:'static_pressure_residual_pa'};
        ctx.side.insertAdjacentHTML('beforeend',`<div class="mv-detail"><strong>${c.scene==='hvac'?'原始残差与最终设定':'模型请求 → 安全投影后'}</strong><div class="mv-action-row mv-action-head"><span>动作</span><span>${c.scene==='hvac'?'请求残差':'请求'}</span><span>${c.scene==='hvac'?'最终设定':'投影后'}</span></div>${actionEntries.map(([k,v])=>`<div class="mv-action-row"><span title="${esc(k)}">${esc(names[k]||k)}</span><span>${fmt(requested[residualKeys[k]||k],Math.abs(requested[residualKeys[k]||k])<.01?6:3)}</span><b>${fmt(v,Math.abs(v)<.01?6:3)}</b></div>`).join('')||'<p>本版本未返回结构化动作；请通过下方模型输出按钮核验。</p>'}</div>`);
        ctx.side.insertAdjacentHTML('beforeend',`<div class="mv-tags"><span class="mv-tag ${inf.policy_loaded?'pass':'pending'}">权重${inf.policy_loaded?'已重载':'待核验'}</span><span class="mv-tag pending">现场${inf.production_admitted?'已准入':'待准入'}</span></div>`);
        ctx.side.insertAdjacentHTML('beforeend',`<details><summary>来源时间与展示口径</summary><p>推理来源时间 ${esc(inf.timestamp||'未返回')}。几何、数量和动画为结构示意；动作仅为本次模型证据，不驱动现场设备。</p></details>`);
        ctx.status.textContent=inf.timestamp?`模型证据时间 ${inf.timestamp}`:'模型证据已返回';
        ctx.caption.textContent='结构示意 · 工程/公开回放证据 · 点选设备可查看部件名称';
        if(d.business_metrics){const b=d.business_metrics,find=prefix=>Object.entries(b).find(([k,v])=>k.startsWith(prefix)&&valid(v));const values=[['成本改善',find('cost_reduction_')],['能耗改善',find('energy_reduction_')],['峰值改善',find('peak_reduction_')],['碳改善',find('carbon_reduction_')]].filter(([,p])=>p);ctx.foot.innerHTML=values.map(([n,p])=>`<span>${esc(n)} <b>${fmt(p[1],2)}%</b></span>`).join('')+'<span>独立盲测 · 对照与范围见下方业务证据</span>';}
        return;
      }
      if(c.kind==='spatial'||c.kind==='energy'){
        const d=window.PortDemoData?(await window.PortDemoData.load(refresh)).snapshot:await json('/api/public-port-demo/snapshot',refresh);if(active!==ctx)return;const o=d.operating;
        if(c.kind==='energy'){
          const rows=d.series.map(r=>({...r,demand_kw:r.power_kw,throughput_teu:r.throughput_teu_per_hour}));
          trace(ctx,ctx.main,rows,[{key:'demand_kw',name:'标定负荷',unit:'kW'},{key:'throughput_teu',name:'标定吞吐',unit:'TEU/h'}],'展示源24小时窗口；来源核验以原有网关为准');
          stats(ctx.side,[['24h 标定用电',fmt(o.daily_energy_kwh/1000,1),'MWh · 工程派生'],['24h 工程碳排',fmt(o.daily_carbon_kg/1000,2),'tCO₂e · 同窗电力范围'],['单位吞吐能耗',fmt(o.energy_kwh_per_teu,3),'kWh/TEU · 同窗分母'],['单位吞吐碳排',fmt(o.carbon_kg_per_teu,3),'kgCO₂e/TEU · 非审计值']]);
        }else{
          stats(ctx.side,[['作业泊位',`${o.berths_busy} / ${o.berths_total}`,'个 · 资源约束推演'],['岸桥分配',`${o.qc_busy} / ${o.qc_total}`,'台 · 推演配桥'],['AGV 作业映射',`${o.agv_busy} / ${o.agv_total}`,'台 · 负荷比例映射'],['标定功率',fmt(o.power_kw,0),'kW · 工程场景']]);
        }
        note(ctx.side,'点击上方“港口作业服务台”查看泊位计划、设备群、能碳与来源。历史模型与盲测保留原有数据口径。');ctx.status.textContent=`${sourceName(d)} · ${d.source_timestamp}`;ctx.caption.textContent='公开参数标定演示 · 运行指标为资源约束推演/工程派生';
      }else if(['executive','carbon','compliance','environment'].includes(c.kind)){
        if(c.kind==='executive'||c.kind==='environment'){
          const d=window.PortDemoData?(await window.PortDemoData.load(refresh)).snapshot:await json('/api/public-port-demo/snapshot',refresh);if(active!==ctx)return;
          const rows=d.series.map(r=>({...r,demand_kw:r.power_kw,throughput_teu:r.throughput_teu_per_hour})),o=d.operating;
          if(c.kind==='executive'){
            trace(ctx,ctx.main,rows,[{key:'demand_kw',name:'标定负荷',unit:'kW'},{key:'throughput_teu',name:'标定吞吐',unit:'TEU/h'}],'展示源24小时负荷与吞吐窗口');
            stats(ctx.side,[['演示作业泊位',`${o.berths_busy} / ${o.berths_total}`,'个 · 推演占用'],['24h 标定箱量',fmt(o.daily_throughput_teu,0),'TEU · 工程场景'],['岸桥作业分配',`${o.qc_busy} / ${o.qc_total}`,'台 · 资源约束'],['AGV 作业映射',`${o.agv_busy} / ${o.agv_total}`,'台 · 演示工作量']]);note(ctx.side,`${d.profile.terminal||d.profile.name||'当前港口'} · ${sourceName(d)}；下方原有经营证据、历史盲测与现场就绪态保留各自统计范围。`);
          }else{
            trace(ctx,ctx.main,rows,[{key:'wind_speed_mps',name:'风速',unit:'m/s'},{key:'tide_m',name:'潮位',unit:'m'}],'公开历史再分析：风速m/s、潮位m，浪高见侧栏');const last=d.environment;stats(ctx.side,[['风速',fmt(last.wind_speed_mps,2),'m/s · 历史再分析'],['潮位',fmt(last.tide_m,2),'m · 原序列参考面'],['浪高',fmt(last.wave_height_m,2),'m · 历史再分析'],['推演作业航次',fmt(o.working_call_count,0),'航次 · 非真实AIS身份']]);note(ctx.side,'海况是历史再分析，航次是资源约束推演；“作业服务台”提供每泊位计划与来源说明，不把推演时间标作真实ETA。');
          }
          ctx.status.textContent=`${sourceName(d)} · ${d.source_timestamp}`;ctx.caption.textContent='公开参数标定演示 · 非现场产量与实际航次';
        }else{
          const view=await window.PortDemoData.load(refresh);if(active!==ctx)return;
          const a=view.annual,w=view.window,s=view.snapshot,annual=a.available,teu=annual?a.totals.throughput_teu:w.throughput_teu,energy=annual?a.totals.energy_kwh:w.energy_kwh,carbon=annual?a.totals.electricity_carbon_kg:w.electricity_carbon_kg;
          flow(ctx.main,c.kind==='carbon'?[['标定吞吐',`${fmt(scaled(teu,1e-6),3)} 百万 TEU`],['工程电量',`${fmt(scaled(energy,1e-6),2)} GWh`],['加权排放因子',`${valid(carbon)&&energy>0?fmt(carbon/energy,3):'—'} kg/kWh`],['电力碳排',`${fmt(scaled(carbon,1e-3),1)} tCO₂e`]]:[['统一场景源','公开规模 / 工程小时序列'],['分时与分项','小时积分 / 机群功率'],['演示报表',annual?'2025 · 12个月 / 8,760小时':'外部源提供的时间窗口'],['现场核证','凭证、边界与签发人待接入']]);
          stats(ctx.side,[['单位吞吐碳强度',valid(carbon)&&teu>0?fmt(carbon/teu,3):'—','kgCO₂e/TEU · 电力工程估算'],['统计范围',annual?'洋山四期标定年':'外部源序列窗口',annual?'2025 UTC完整年':'不混入内置年度'],['引用公开资料',fmt(s.profile.sources.length,0),'条 · 来源服务台可核验'],['本演示报表','未核证','Scope 1 / 全范围台账另接']]);note(ctx.side,annual?a.boundary:w.boundary||w.reason);ctx.status.textContent=`${view.source_mode} · ${s.source_timestamp}`;ctx.caption.textContent='电力工程测算 · 下方提供月度与分时明细 · 原有现场报表入口保留';
        }
      }else if(c.kind==='profiles'){
        const d=await json('/api/multiport/summary',refresh);if(active!==ctx)return;const profiles=d.profiles||[];
        const portNames={CNSHA:'上海',SGSIN:'新加坡',USLAX:'洛杉矶',REFERENCE:'工程参考'};const names=p=>`${portNames[p.port_code]||p.port_code} · ${p.environment_version.replace('port_ops_','')}`;const nodes=[{id:'hub',name:'统一端口契约',x:0,y:0,symbolSize:74,itemStyle:{color:'#2e7772'}}];profiles.forEach((p,i)=>{const a=i/profiles.length*Math.PI*2;nodes.push({id:p.profile_id,name:names(p),x:Math.cos(a)*230,y:Math.sin(a)*115,symbolSize:35,itemStyle:{color:colors[i%colors.length]}});});
        const graph=chart(ctx,ctx.main,{tooltip:{trigger:'item',formatter:p=>esc(p.data.name||'画像契约')},series:[{type:'graph',layout:'none',data:nodes,links:profiles.map(p=>({source:'hub',target:p.profile_id})),roam:false,label:{show:true,position:'bottom',fontSize:10,color:'#bdd7df'},lineStyle:{color:'#58857f',width:1.3,curveness:.1},emphasis:{focus:'adjacency'}}]},`${profiles.length}份端口画像的逻辑关系网络，非地理位置或生产部署`);
        const show=p=>{stats(ctx.side,[['注册画像',fmt(profiles.length,0),'非已部署港区数'],['端口代码',p?.port_code||'—','端口画像'],['训练字段',fmt(p?.required_for_training?.length,0),'注册表要求'],['现场必填',fmt(p?.required_for_site_claim?.length,0),'需授权来源替换']]);note(ctx.side,`${p?.name||''} · ${p?.environment_version||''}`);ctx.side.insertAdjacentHTML('beforeend',`<div class="mv-detail"><strong>现场必填字段</strong><div class="mv-tags">${(p?.required_for_site_claim||[]).map(v=>`<span class="mv-tag">${esc(v)}</span>`).join('')}</div></div>`);note(ctx.side,'节点来自端口画像注册表。连线表示共用契约，不代表部署、迁移训练完成或现场联网。');};show(profiles.find(p=>p.port_code==='CNSHA')||profiles[0]);graph?.on('click',p=>{const profile=profiles.find(v=>v.profile_id===p.data.id);if(profile)show(profile);});ctx.status.textContent=`${profiles.length} 份注册画像 · 现场部署待接入`;
      }else if(c.kind==='apps'){
        const d=await json('/api/app_center/overview',refresh);if(active!==ctx)return;const routes=d.platform_support?.rest_apis||[],groups={};routes.forEach(r=>{const name=r.path.split('/')[2]||'其他';groups[name]=(groups[name]||0)+1;});const items=Object.entries(groups).sort((a,b)=>b[1]-a[1]).slice(0,10);
        chart(ctx,ctx.main,{grid:{left:95,right:35,top:10,bottom:25},xAxis:{...axis,type:'value',minInterval:1},yAxis:{...axis,type:'category',inverse:true,data:items.map(v=>v[0]),splitLine:{show:false}},series:[{type:'bar',data:items.map(v=>v[1]),barMaxWidth:14,label:{show:true,position:'right',color:'#abc7d3'},itemStyle:{color:'#75c8ba'}}]},'已注册接口按业务域计数，前10个域');stats(ctx.side,[['REST 路由',fmt(routes.length,0),'运行时注册'],['接口业务域',fmt(Object.keys(groups).length,0),'按路径分组'],['现场 App','待接入','不推断现场部署'],['文档入口','OpenAPI','下方原有文档与复制按钮']]);ctx.status.textContent='已读取运行时路由注册表';
      }else if(c.kind==='monitoring'){
        const d=await json('/api/v3/monitoring/evidence',refresh);if(active!==ctx)return;const a=d.current_analysis?.anomaly||{},drift=d.current_analysis?.drift||{},bins=drift.bins||[];
        if(bins.length)chart(ctx,ctx.main,{legend:{top:0,textStyle:{color:'#8eacbd'}},xAxis:{...axis,type:'category',data:bins.map((b,i)=>valid(b.lo)?`${fmt(b.lo,0)}~${fmt(b.hi,0)}`:String(i+1))},yAxis:{...axis,type:'value',name:'样本比例'},series:[{name:'基线窗口',type:'bar',data:bins.map(b=>b.p_ref),barMaxWidth:16},{name:'近期窗口',type:'bar',data:bins.map(b=>b.p_cur),barMaxWidth:16}]},'基线与近期窗口分箱比例，来自监控分析返回');else ctx.main.innerHTML='<div class="mv-empty"><b>分布窗口证据不足</b><span>请先用下方 PSI 漂移按钮读取结果。</span></div>';
        stats(ctx.side,[['PSI',fmt(drift.psi,4),'基线 / 近期窗口'],['异常样本',fmt(a.anomaly_count,0),'本次算法扫描'],['分析样本',fmt(a.sample_count,0),'回放输入'],['现场事件','待接入','不等同实际事故']]);note(ctx.side,d.boundary?.reason||'公开数据校准回放，现场告警与工单另行接入。');ctx.status.textContent='当前窗口分析结果 · 校准回放';
      }else{
        const d=await json('/api/v3/overview',refresh);if(active!==ctx)return;
        if(c.kind==='architecture')flow(ctx.main,[['港口设备','岸桥 / 场桥 / AGV / 能源'],['数据与孪生',`${fmt(d.dataset?.rows,0)} 行公开目标域数据`],['模型与安全',`${fmt(d.algorithms?.length,0)} 类算法 / 控制基线`],['业务与运营',`${fmt(d.capabilities?.length,0)} 个业务域 · 现场待接入`]]);
        if(c.kind==='advantage')advantage(ctx,ctx.main,d);
        if(c.kind==='training')split(ctx,ctx.main,d);
        if(c.kind==='domains')matrix(ctx.main,(d.capabilities||[]).map(v=>[v.name,v.depth?.model_output_available?'模型输出可核验 · 现场待接入':'工程契约 · 现场待接入',v.depth?.model_output_available===true]));
        if(c.kind==='coverage'){
          const coverage=await json('/api/v3/runtime/coverage',refresh);if(active!==ctx)return;matrix(ctx.main,(coverage.scenarios||[]).map(s=>[s.name.split(' / ')[0],s.state==='contract_only'?'契约待实测':s.state==='fail_closed_covered'?'软件失效安全覆盖':'软件场景覆盖',s.state!=='contract_only']));
        }
        const bc=d.business_domain_coverage||{};
        const rows=c.kind==='domains'?[['业务域',fmt(bc.domain_count,0),'后端能力目录'],['模型优化输出',fmt(bc.learned_optimizer_output_count,0),'具学习优化器'],['现场替换需求',fmt(bc.site_replacement_required_count,0),'域 · 接口/验收待补'],['现场就绪',fmt(bc.production_ready_count,0),'域 · 后端明确统计']]:[['目标域数据',fmt(d.dataset?.rows,0),'行 · 公开与工程派生'],['算法 / 基线',fmt(d.algorithms?.length,0),'类 · 正式证据'],['正式训练记录',fmt(d.evidence?.formal_run_count,0),'次 · 持久化留痕'],['生产控制','未授予','需独立现场准入']];stats(ctx.side,rows);
        if(c.kind==='advantage')note(ctx.side,'柱形为后端相对改善率；负值表示该项退化。白色横线为三种子均值的 95% bootstrap 区间。总成本与等量吞吐成本口径不同。');
        note(ctx.side,'公开数据离线与工程场景证据；现场遥测、财务收益和设备控制权须独立验收。');
        ctx.status.textContent=`证据版本 ${d.version||'—'} · 目标域数据哈希 ${String(d.dataset?.sha256||'—').slice(0,12)}`;
      }
    }catch(error){if(active!==ctx)return;ctx.status.textContent=`数据暂不可用 · ${error.name==='AbortError'?'请求超时':error.message} · 可点击刷新重试`;if(!['spatial','facility'].includes(c.kind))ctx.main.innerHTML='<div class="mv-empty"><b>本次数据尚未取得</b><span>保留原有操作入口，点击“刷新视图”重新核验。</span></div>';stats(ctx.side,[['本次证据','未返回','不补造数据'],['现场状态','待接入','保持原有授权边界']]);}
    finally{if(active===ctx){ctx.refresh.disabled=false;ctx.board.dataset.loadMs=(performance.now()-ctx.loadStarted).toFixed(1);}}
  }
  function dispose(){if(!active)return;active.spatial?.dispose();active.charts.forEach(c=>c.dispose());active.clean.forEach(f=>f());active.chartClean.forEach(f=>f());active.board.remove();active=null;}
  function mount(id,root){
    const c=configs[id];if(!c||!root)return;if(active?.id===id)return;dispose();
    document.body.classList.add('mv-desktop');
    const board=document.createElement('section');board.className='mv-workspace';board.setAttribute('data-i18n-skip','');board.dataset.visualModule=id;board.setAttribute('aria-label',c.title+'可视工作台');
    const anchor=`mv-original-${id}`;let original=Array.from(root.children).find(el=>!['SCRIPT','STYLE','LINK'].includes(el.tagName));
    if(!original&&id==='exec-wrap'){
      original=document.createElement('div');original.className='card';
      original.innerHTML='<div class="title">执行安全工作台</div><p class="small">先检查策略评测与南向安全边界；设备命令仍使用原有暂存、异人确认、回执与回滚流程。</p><div class="row" style="gap:12px;margin-top:16px"><a class="btn" href="#exec-section">打开原有执行与审批工作区 →</a><a class="btn" href="#strategy-exec-module">查看策略与仿真证据 →</a></div>';
      root.append(original);
    }
    if(original){if(!original.id)original.id=anchor;else board.dataset.originalId=original.id;}
    board.innerHTML=`<div class="mv-head"><div><div class="mv-eyebrow">${esc(c.en)}</div><h2>${esc(c.title)}</h2><p>${esc(c.sub)}</p></div><div class="mv-tools"><span class="mv-pill">公开 / 工程证据 · 现场待接入</span><button type="button" data-mv-action="refresh">刷新视图</button><a href="#${esc(board.dataset.originalId||anchor)}" data-mv-original>操作与证据 ↓</a></div></div><div class="mv-body"><div class="mv-panel"><div class="mv-panel-title"><span>${esc(c.chart)}</span><small>${['facility','spatial'].includes(c.kind)?'INTERACTIVE 3D':'EVIDENCE VIEW'}</small></div><div class="mv-main-content"></div><div class="mv-caption"><span class="mv-caption-copy">${['facility','spatial'].includes(c.kind)?'结构与运动示意 · 数据以右侧证据为准':'随数据来源展示 · 缺失值保持为空'}</span><code>${esc(c.endpoint?'/api/v3/modules/'+c.endpoint+'/evidence':'原有接口 / 来源证据')}</code></div></div><aside class="mv-panel mv-side"><h3>${esc(c.side)}</h3><div class="mv-side-content"><div class="mv-empty">读取模块证据…</div></div></aside></div><div class="mv-bottom"><span class="mv-status" role="status">读取当前模块数据…</span><div class="mv-bottom-values"></div></div>`;
    root.prepend(board);const ctx={id,root,board,config:c,main:board.querySelector('.mv-main-content'),side:board.querySelector('.mv-side-content'),status:board.querySelector('.mv-status'),caption:board.querySelector('.mv-caption-copy'),foot:board.querySelector('.mv-bottom-values'),refresh:board.querySelector('[data-mv-action="refresh"]'),charts:[],clean:[],chartClean:[],spatial:null};active=ctx;
    if(['facility','spatial'].includes(c.kind))sceneMarkup(ctx);else ctx.main.innerHTML='<div class="mv-empty">读取可视化数据…</div>';
    board.addEventListener('click',e=>{const b=e.target.closest('[data-mv-action]');if(b){const a=b.dataset.mvAction;if(a==='refresh'){ctx.charts.forEach(c=>c.dispose());ctx.charts=[];ctx.chartClean.forEach(f=>f());ctx.chartClean=[];if(['spatial','facility'].includes(c.kind)&&!ctx.spatial)spatial(ctx);load(ctx,true);}if(a==='reset'){ctx.spatial?.reset();board.querySelector('[data-mv-action="top"]')?.setAttribute('aria-pressed','false');}if(a==='top')b.setAttribute('aria-pressed',String(ctx.spatial?.top()||false));if(a==='motion'){const p=ctx.spatial?.pause();b.textContent=p?'继续示意动画':'暂停示意动画';b.setAttribute('aria-pressed',String(!!p));}}if(e.target.closest('[data-mv-original]')){e.preventDefault();e.stopPropagation();original?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});}},true);
    if(['pipeline','services','copilot','governance'].includes(c.kind)){let scheduled=false;const observer=new MutationObserver(list=>{if(list.every(r=>board.contains(r.target))||scheduled)return;scheduled=true;queueMicrotask(()=>{scheduled=false;if(active===ctx)redrawDom(ctx);});});observer.observe(root,{subtree:true,childList:true,characterData:true});ctx.clean.push(()=>observer.disconnect());}
    requestAnimationFrame(()=>{if(active===ctx)window.scrollTo({top:0,behavior:'instant'});});
    ctx.loadStarted=performance.now();load(ctx);window.dispatchEvent(new CustomEvent('port-visual-workspace-ready',{detail:{id}}));
  }
  function route(){const id=document.body.dataset.moduleView;if(id==='home-hero'||preservedModelPages.has(id)){dispose();document.body.classList.remove('mv-desktop');return;}if(configs[id])mount(id,document.getElementById(id));}
  function init(){
    if(document.getElementById('home-hero')){window.addEventListener('port-module-shown',route);route();return;}
    const path=location.pathname;let id,surface;
    if(path==='/rl-panel')return;
    if(path.startsWith('/v3')){id='standalone-v3';surface='v3';}else if(path.includes('rl_future')){id='standalone-future';surface='future';}else if(path==='/integration-hub'){id='standalone-hub';surface='hub';}else if(path==='/ops-copilot'){id='standalone-copilot';surface='copilot';}
    if(id){document.body.dataset.mvSurface=surface;mount(id,document.querySelector('main'));}
    if(surface==='future'){
      const guards=document.getElementById('guardList'),panel=document.getElementById('futureEvidenceRadar');
      if(guards&&panel){
        const update=()=>{const items=Array.from(guards.querySelectorAll('li'));
          panel.innerHTML=items.length?items.map(item=>`<div class="${item.classList.contains('pass')?'pass':''}">${esc(item.textContent.split(' · ')[0])}<span>${item.classList.contains('pass')?'通过':item.classList.contains('block')?'阻断':'等待本次推演'}</span></div>`).join(''):`<div>本次推演<span>${esc(guards.textContent)}</span></div>`;};
        const observer=new MutationObserver(update);observer.observe(guards,{childList:true,subtree:true,characterData:true});update();active?.clean.push(()=>observer.disconnect());
      }
    }
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
  window.addEventListener('pagehide',()=>{dispose();});
  window.addEventListener('pageshow',event=>{if(event.persisted)init();});
})();
