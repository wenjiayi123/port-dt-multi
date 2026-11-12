/* Read-only overview. All displayed values come from existing backend contracts. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const home = $('home-hero');
  if (!home || !home.classList.contains('hd-home')) return;
  const put = (id, value) => { const node = $(id); if (node) node.textContent = value; };
  const number = value => value !== null && value !== undefined && value !== '' && typeof value !== 'boolean' && Number.isFinite(Number(value)) ? Number(value) : null;
  const fmt = (value, digits = 0) => number(value) === null ? '—' : Number(value).toLocaleString('zh-CN', {maximumFractionDigits:digits});
  const active = () => !document.hidden && (document.body.dataset.moduleView || 'home-hero') === 'home-hero';
  const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
  const state = {layout:null, frame:null, provenance:null, assets:[], environment:null, portCall:null, coordination:null, paused:false, busy:false, energyBusy:false, layer:'overview', lastGood:0, loads:[], throughputs:[], history:[], historyIndex:-1, scene:null, theme:'day', labels:true, tour:false, selection:{key:'quay',index:null}};
  let energyChart, serviceChart, initialization, sceneLibrary, libraryAttempt=0, serviceToken=0, servicesPending, layoutRequest;
  const safe = value => String(value ?? '—').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

  async function json(path) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(path, {cache:'no-store', signal:controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } finally { clearTimeout(timer); }
  }
  function bootstrap() {
    return layoutRequest ||= json('/api/portviz/bootstrap').then(layout=>{
      state.layout=layout;state.presentationMode=layout.meta?.source_mode==='public_parameter_calibrated_demo';return layout;
    }).catch(error=>{layoutRequest=null;throw error;});
  }
  const energyUrl=id=>state.presentationMode?`/api/public-port-demo/asset-series/${encodeURIComponent(id)}`:`/api/telemetry/recent/${encodeURIComponent(id)}`;

  function initHeader() {
    const title = document.querySelector('.hero-title');
    // Keep a single brand and the existing menu controls/event bindings.
    if (title) {
      title.textContent = '港口 AI 运营中枢';
      const small = document.createElement('small'); small.textContent = 'TWINPORT AI'; title.append(small);
      const mark = document.createElement('span'); mark.className = 'hd-brand-mark'; mark.setAttribute('aria-hidden','true');
      mark.innerHTML = '<svg viewBox="0 0 32 32" fill="none"><path d="M5 24h22M7 19h18l-3 5H10l-3-5ZM15 6h3v13h-3zM7 10h18M10 7v9M23 10v6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      title.before(mark);
      const toggle = document.createElement('button'); toggle.type = 'button'; toggle.className = 'hd-menu-toggle'; toggle.textContent = '功能菜单 ☰'; toggle.setAttribute('aria-expanded','false'); toggle.setAttribute('aria-controls','panel-nav-primary');
      toggle.addEventListener('click', () => { const open = document.body.classList.toggle('hd-menu-open'); toggle.setAttribute('aria-expanded', String(open)); });
      title.parentElement.append(toggle);
    }
    const clock = () => put('hd-clock', new Date().toLocaleTimeString('en-GB', {timeZone:'Asia/Kuala_Lumpur', hour12:false}));
    clock(); setInterval(() => { if (active()) clock(); }, 1000);
  }

  function spark(values, id) {
    if (values.length < 2) return;
    const low = Math.min(...values), high = Math.max(...values), span = high - low || 1;
    $(id).setAttribute('d', values.map((v,i) => `${i ? 'L' : 'M'}${i / (values.length - 1) * 100},${33 - (v - low) / span * 25}`).join(' '));
  }

  function bars(id, list) {
    const root = $(id); root.replaceChildren();const items=list||[];
    if(items.length>32){root.style.cssText='display:grid;grid-template-columns:repeat(12,2px);height:22px;gap:2px';for(let i=0;i<24;i++){const group=items.slice(Math.floor(i*items.length/24),Math.floor((i+1)*items.length/24)),busy=group.filter(v=>v.busy===true).length,bar=document.createElement('i');bar.style.cssText='width:2px;height:9px';bar.classList.toggle('on',busy>0);bar.title=`设备簇 ${i+1}：${busy}/${group.length} 作业`;root.append(bar);}return;}
    root.style.cssText='';items.forEach(item => { const bar = document.createElement('i'); bar.classList.toggle('on', item.busy === true); root.append(bar); });
  }

  function renderFrame(frame,record=true) {
    const metrics = frame.metrics || {};
    put('hd-load', fmt(metrics.base_load_kw)); put('hd-throughput', fmt(metrics.throughput_teu,1));
    const qc = Array.isArray(frame.qc) ? frame.qc : null, yc = Array.isArray(frame.yc) ? frame.yc : null;
    put('hd-qc-busy', qc ? qc.filter(x => x.busy === true).length : '—'); put('hd-qc-total', `/ ${qc ? qc.length : '—'} 台`);
    put('hd-yc-busy', yc ? yc.filter(x => x.busy === true).length : '—'); put('hd-yc-total', `/ ${yc ? yc.length : '—'} 台`);
    bars('hd-qc-bars', qc); bars('hd-yc-bars', yc);
    put('hd-asset-qc', qc ? qc.length : '—'); put('hd-asset-yc', yc ? yc.length : '—');
    put('hd-asset-agv', Array.isArray(frame.agv) && Array.isArray(frame.tr) ? frame.agv.length + frame.tr.length : '—');
    put('hd-asset-vessel', Array.isArray(frame.vessels) ? frame.vessels.length : '—');
    const source = String(frame.source_timestamp || '');
    put('hd-frame-time', source ? `${source.replace('T',' ').replace('Z','')} · 来源时间` : '来源时间未提供');
    const measured = frame.meta?.measured_entity_tracks === true;
    put('hd-scene-mode', measured ? '实体轨迹适配 · 配置布局' : frame.meta?.source_mode==='public_parameter_calibrated_demo'?'洋山四期 · 公开参数标定演示':'数据集投影 · 示意布局');
    if (record && number(metrics.base_load_kw) !== null) state.loads.push(Number(metrics.base_load_kw));
    if (record && number(metrics.throughput_teu) !== null) state.throughputs.push(Number(metrics.throughput_teu));
    state.loads = state.loads.slice(-30); state.throughputs = state.throughputs.slice(-30);
    spark(state.loads, 'hd-load-spark'); spark(state.throughputs, 'hd-throughput-spark');
    state.scene?.update(frame);
    drawMinimap(frame);
    if (!$('hd-scene-inspector').hidden) inspect(state.selection.key, false, state.selection.index);
  }

  function streamStatus(text, type = '') {
    const node = $('hd-stream-status'); node.className = `hd-stream-status ${type}`;
    node.replaceChildren(document.createElement('i'), document.createTextNode(text));
  }

  async function loadFrame() {
    if (state.busy || state.paused || !active()) return;
    state.busy = true;
    try {
      const frame = await json('/api/portviz/stream');
      if (!Array.isArray(frame.agv) || !Array.isArray(frame.qc) || !Array.isArray(frame.yc) || !Array.isArray(frame.tr)) throw new Error('回放帧结构不完整');
      // A pause/navigation during a request must not advance the displayed frame.
      if (state.paused || !active()) return;
      state.frame = frame; state.lastGood = Date.now();
      state.history.push(frame); if(state.history.length>120)state.history.shift();
      state.historyIndex=state.history.length-1;syncHistory();renderFrame(frame);
      home.dataset.frameTimestamp = frame.source_timestamp || String(frame.ts || '');
      if($('hd-data-message').textContent.startsWith('态势数据'))put('hd-data-message','');
      streamStatus(frame.meta?.measured_entity_tracks === true ? '轨迹适配' : '回放中');
    } catch (error) {
      if (!state.paused && active()) {
        streamStatus(state.lastGood ? '数据中断 · 保留末帧' : '回放未连接', 'error');
        put('hd-data-message', '态势数据未更新，可点击右上角刷新重试。');
      }
    } finally { state.busy = false; }
  }

  async function loadProvenance() {
    try {
      const p = await json('/api/system/provenance?detail=summary'); state.provenance = p;
      put('hd-site-status', p.telemetry?.measured === true ? '已接入实测' : '待接入');
      put('hd-authority', p.runtime_policy?.production_authority === false ? '未授权' : '待核验');
      const sourcePill = document.querySelector('.hd-source-pill');
      if(sourcePill&&!sourcePill.dataset.ppdEntry)sourcePill.replaceChildren(document.createElement('i'), document.createTextNode(p.telemetry?.measured === true ? '现场遥测已接入' : '公开数据回放'));
    } catch (_) {
      put('hd-site-status','来源不可读'); put('hd-authority','待核验');
      put('hd-data-message','来源核验暂不可用，请刷新重试。');
    }
  }

  async function loadCoverage() {
    try {
      const data = await json('/api/exec_cockpit/summary');
      const covered = number(data.software_scenario_coverage?.covered), total = number(data.software_scenario_coverage?.total);
      if (covered === null || total === null || total <= 0 || covered < 0 || covered > total) throw new Error('coverage unavailable');
      put('hd-coverage-count', `${covered}/${total}`); put('hd-coverage-ready', covered); put('hd-coverage-rest', total - covered);
      $('hd-coverage-ring').style.setProperty('--coverage', `${270 * covered / total}deg`);
      $('hd-coverage-ring').setAttribute('aria-label', `离线软件场景覆盖 ${covered}/${total}，现场准入另行核验`);
    } catch (_) {
      put('hd-coverage-count','—'); put('hd-coverage-ready','—'); put('hd-coverage-rest','—');
      $('hd-coverage-ring').style.setProperty('--coverage','0deg');
    }
  }

  const sourceTime = row => row.source_ts || row.ts;
  async function loadEnergy() {
    const asset = $('hd-energy-asset').value;
    if (!asset || state.energyBusy || !active()) return;
    state.energyBusy = true;
    try {
      const rows = await json(energyUrl(asset));
      if (asset !== $('hd-energy-asset').value) return;
      const points = (Array.isArray(rows) ? rows : rows.points || rows.series || []).filter(p => number(p.kW) !== null && Number.isFinite(Date.parse(sourceTime(p))));
      if (!points.length) throw new Error('没有有效负荷记录');
      const last = points.at(-1); put('hd-energy-value', fmt(last.kW));
      if(asset==='shore-01')put('hd-service-energy',fmt(last.kW));
      const measured = points.every(p => p.measured === true);
      document.querySelector('.hd-tiny-tag').textContent = measured ? '现场实测' : state.presentationMode?'工程分摊':'校准回放';
      const firstTime = new Date(sourceTime(points[0])).toISOString(), lastTime = new Date(sourceTime(last)).toISOString();
      const range = firstTime.slice(0,10) === lastTime.slice(0,10)
        ? `${firstTime.slice(11,16)} — ${lastTime.slice(11,16)} UTC`
        : `${firstTime.slice(5,16).replace('T',' ')} — ${lastTime.slice(5,16).replace('T',' ')} UTC`;
      put('hd-energy-period', range);
      const root = $('hd-energy-chart'); root.setAttribute('aria-label', `${$('hd-energy-asset').selectedOptions[0].textContent}，${measured ? '实测' : state.presentationMode?'工程分摊':'校准回放'}负荷，${range}，最新 ${fmt(last.kW)} kW`);
      if (window.echarts) {
        energyChart ||= window.echarts.init(root, null, {renderer:'svg'});
        energyChart.setOption({animation:!motionQuery.matches, grid:{top:14,bottom:6,left:2,right:3}, xAxis:{type:'category',boundaryGap:false,data:points.map(p => sourceTime(p)),show:false}, yAxis:{type:'value',scale:true,splitNumber:2,axisLabel:{show:false},splitLine:{lineStyle:{color:'#243946',type:'dashed'}}}, tooltip:{trigger:'axis',confine:true,backgroundColor:'#102c3a',borderColor:'#3b626b',textStyle:{fontSize:10,color:'#c9e4ee'},formatter:params => { const p = points[params[0].dataIndex]; return `${new Date(sourceTime(p)).toISOString().replace('T',' ').slice(0,19)} UTC<br/>${fmt(p.kW,1)} kW · ${measured ? '实测' : state.presentationMode?'工程分摊':'校准回放'}`; }}, series:[{type:'line',data:points.map(p => Number(p.kW)),smooth:.3,symbol:'none',lineStyle:{color:'#67d8c6',width:2},areaStyle:{color:{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{offset:0,color:'#4dbaa642'},{offset:1,color:'#4dbaa600'}]}},markPoint:{symbol:'circle',symbolSize:5,data:[{coord:[points.length-1,Number(last.kW)]}],itemStyle:{color:'#9becda'}}}]},true);
      }
    } catch (_) {
      if (asset === $('hd-energy-asset').value) {
        put('hd-energy-value','—'); put('hd-energy-period','曲线不可用 · 刷新重试');
        $('hd-energy-chart').setAttribute('aria-label','负荷曲线暂不可用'); energyChart?.clear();
      }
    } finally {
      state.energyBusy = false;
      if (asset !== $('hd-energy-asset').value) loadEnergy();
    }
  }

  async function loadAssets() {
    try {
      await bootstrap();
      const rows = await json(state.presentationMode?'/api/public-port-demo/assets':'/api/assets'); const select = $('hd-energy-asset');
      if (!Array.isArray(rows) || !rows.length) throw new Error('assets unavailable');
      state.assets=rows;const previous = select.value;
      select.replaceChildren(...rows.filter(row => row.id || row.asset_id).map(row => { const option = document.createElement('option'); option.value = row.id || row.asset_id; option.textContent = row.label || option.value; return option; }));
      if ([...select.options].some(o => o.value === previous)) select.value = previous;
      await loadEnergy();
    } catch (_) { put('hd-energy-period','设备来源不可用'); }
  }

  function syncHistory() {
    const slider=$('hd-history-slider');slider.max=String(Math.max(0,state.history.length-1));slider.value=String(Math.max(0,state.historyIndex));slider.disabled=state.history.length<2;
    put('hd-history-count',`${state.history.length} 帧`);
    put('hd-history-label',state.historyIndex<state.history.length-1?'历史留帧':'会话回放');
    $('hd-history-latest').classList.toggle('active',state.historyIndex===state.history.length-1);
  }
  function pause(value) {
    state.paused=value;
    if(value)stopTour();
    const button=$('hd-play');button.textContent=value?'▶':'Ⅱ';button.setAttribute('aria-label',value?'继续态势回放':'暂停态势回放');button.setAttribute('aria-pressed',String(value));button.title=value?'继续回放':'暂停回放';streamStatus(value?'回放已暂停':'连接回放',value?'paused':'');
    if(!value)loadFrame();
  }
  function drawMinimap(frame) {
    const l=state.layout;if(!l)return;const sx=160/l.world.W,sy=90/l.world.H;
    const blocks=(l.yards||[]).map(b=>`<rect x="${b.x*sx}" y="${b.y*sy}" width="${b.w*sx}" height="${b.h*sy}" rx=".7" fill="#527c80"/>`).join('');
    const lines=(l.lanes||[]).map(lane=>`<polyline points="${lane.map(p=>`${p.x*sx},${p.y*sy}`).join(' ')}" stroke="#3c6776" fill="none" stroke-width="1"/>`).join('');
    const vehicles=(frame?.tr||[]).slice(0,16).map(p=>`<circle cx="${p.x*sx}" cy="${p.y*sy}" r=".8" fill="#82ddc8"/>`).join('');
    $('hd-minimap-svg').innerHTML=`<rect width="160" height="90" fill="#122c3a"/><path d="M0 0H160V17H0Z" fill="#16435a"/>${blocks}${lines}${vehicles}`;
  }
  function inspect(key, move = true, index = null) {
    state.selection={key,index};
    const frame = state.frame || {}, layout = state.layout || {};
    const details = {
      quay:['前沿泊位',`当前回放包含 ${Array.isArray(frame.qc) ? frame.qc.length : '—'} 台岸桥、${Array.isArray(frame.vessels) ? frame.vessels.length : '—'} 艘船舶。岸桥颜色表示当前帧繁忙状态；船型与布局为示意。`],
      yard:['集装箱堆场',`布局包含 ${Array.isArray(layout.yards) ? layout.yards.length : '—'} 个堆场区块；回放帧包含 ${Array.isArray(frame.yc) ? frame.yc.length : '—'} 台场桥。箱体排列为场景表达，不作为实际库存统计。`],
      transport:['水平运输网络',`当前帧包含 ${Array.isArray(frame.agv) ? frame.agv.length : '—'} 台 AGV、${Array.isArray(frame.tr) ? frame.tr.length : '—'} 台集卡。车辆位置来自后端数据投影；帧间采用平滑插值。`],
      energy:['岸电与储能设施','岸电接电箱、储能舱和光伏顶棚为工程示意。设备曲线来自现有校准回放；图上的连接线不代表实际送电。'],
      gate:['闸口与铁路联运','闸口查验、车道、铁路及集装箱平车为设施示意。预约、通行与列车计划需由现场 TOS/闸口/铁路系统接入。'],
      vessel:['集装箱船 VIS-V001','船型为示意模型；场景帧提供几何长度与作业进度，未提供实船船名、IMO、吃水和 ETA。']
    };
    const selected = [...(details[key] || details.quay)];let metrics=[];
    if(key==='quay') {
      if(index!==null){selected[0]=`岸桥 QC-${String(index+1).padStart(2,'0')}`;metrics=[['回放作业态',frame.qc?.[index]?.busy===true?'作业中':'待命'],['小车位置',`${fmt((number(frame.qc?.[index]?.trolley)||0)*100)}%`]];}
      else metrics=[['岸桥',`${frame.qc?.length??'—'} 台`],['繁忙',`${frame.qc?.filter(p=>p.busy).length??'—'} 台`]];
    } else if(key==='yard') {
      if(index!==null)selected[0]=`堆场区块 Y-${String(index+1).padStart(2,'0')}`;
      metrics=[['场桥',`${frame.yc?.length??'—'} 台`],['繁忙',`${frame.yc?.filter(p=>p.busy).length??'—'} 台`]];
    } else if(key==='transport') metrics=[['AGV',`${frame.agv?.length??'—'} 台`],['集卡',`${frame.tr?.length??'—'} 台`]];
    else if(key==='energy') metrics=[['设备数据','校准回放'],['现场接电','待确认']];
    else if(key==='gate') metrics=[['闸口车流','未接入'],['铁路计划','未接入']];
    else if(key==='vessel') {const v=frame.vessels?.[index??0];metrics=[['几何长度',v?`${fmt(v.len)} 布局单位`:'—'],['回放进度',v?`${fmt((number(v.progress)||0)*100)}%`:'—']];}
    put('hd-inspector-title',selected[0]); put('hd-inspector-body',selected[1]); $('hd-scene-inspector').hidden = false;
    $('hd-inspector-metrics').innerHTML=metrics.map(([k,v])=>`<span><small>${safe(k)}</small><b>${safe(v)}</b></span>`).join('');
    $('hd-inspector-link').href={energy:'#sbess-section',gate:'#deep-section',vessel:'#ext-section'}[key]||'#realtime-section';
    if (move) state.scene?.focus(key);
  }

  async function loadServices(force=false) {
    if(servicesPending)return servicesPending;
    servicesPending=(async()=>{
      await bootstrap();
      const results=await Promise.allSettled([json('/api/v3/external-signals/evidence'),json('/api/v3/port-call/readiness'),json('/api/v3/end-to-end-coordination/readiness'),state.presentationMode?json('/api/public-port-demo/snapshot'):Promise.resolve(null)]);
      state.environment=results[0].status==='fulfilled'?results[0].value:null;
      state.portCall=results[1].status==='fulfilled'?results[1].value:null;
      state.coordination=results[2].status==='fulfilled'?results[2].value:null;
      state.presentation=results[3].status==='fulfilled'?results[3].value:null;
      const env=state.presentationMode?state.presentation?.environment:state.environment?.timeline?.[0];
      put('hd-weather-state',env?'公开历史海况':'来源不可用');put('hd-weather-wind',fmt(env?.wind_speed_mps,1));put('hd-weather-tide',fmt(env?.tide_m,2));put('hd-weather-wave',fmt(env?.wave_height_m,2));
      put('hd-berth-state',!state.portCall?'核验不可用':state.portCall.boundary?.live_data_verified===true?'现场源已验证':'现场待接入');
      put('hd-berth-value',state.portCall?.adapter_status?.live_data_verified===true?'靠泊事件源已验证':'船期 / AIS 待接入');
      if(state.presentationMode){const o=state.presentation?.operating;put('hd-berth-state',o?'公开参数推演':'演示源不可用');put('hd-berth-value',o?`${o.berths_busy} / ${o.berths_total} 泊位 · ${o.working_call_count} 推演航次`:'—');}
      const c=state.coordination?.contract;
      put('hd-coordination-state',!c?'来源不可用':state.coordination?.configured_artifact?.verified===true?'现场证据已验证':'合同已登记');
      put('hd-coordination-value',c?`${c.resource_types.length} 类资源 · ${c.stages.length} 个阶段`:'—');
      const shore=state.assets.find(a=>a.id==='shore-01');
      if(shore)try{const rows=await json(energyUrl('shore-01'));put('hd-service-energy',fmt(Array.isArray(rows)?rows.at(-1)?.kW:null));}catch(_){put('hd-service-energy','—');}
    })().finally(()=>{servicesPending=null;});return servicesPending;
  }

  async function openService(kind) {
    const token=++serviceToken,dialog=$('hd-service-dialog'),body=$('hd-service-dialog-body');
    const title={environment:'气象与海况服务',berth:'船舶与靠泊服务',energy:'岸电与能源服务',coordination:'全链条协同服务'}[kind];
    put('hd-service-dialog-title',title);body.innerHTML='<div class="hd-service-empty">正在读取服务信息…</div>';
    if(!dialog.open)dialog.showModal();
    if(!state.environment&&!state.portCall&&!state.coordination)await loadServices();
    if(token!==serviceToken||!dialog.open)return;
    const source=state.presentationMode?{timeline:state.presentation?.series}:state.environment, sample=state.presentationMode?state.presentation?.environment:source?.timeline?.[0];
    const row=(k,v)=>`<div class="hd-service-detail-row"><span>${safe(k)}</span><strong>${safe(v)}</strong></div>`;
    const note=text=>`<p class="hd-service-boundary">${safe(text)}</p>`;
    let route='#ext-section';
    if(kind==='environment') {
      if(!sample)body.innerHTML=note('环境来源暂不可用，请关闭后刷新首页重试。');
      else {
        body.innerHTML=`<div class="hd-service-env-grid">${[['风速',sample.wind_speed_mps,'m/s'],['潮位',sample.tide_m,'m'],['浪高',sample.wave_height_m,'m'],['气温',sample.ambient_c,'°C']].map(([k,v,u])=>`<div><span>${k}</span><b>${fmt(v,2)} <small>${u}</small></b></div>`).join('')}</div><div class="hd-service-chart" id="hd-service-chart"></div>${note(state.presentationMode?'与洋山演示同一历史时刻的公开再分析序列，非当前现场气象。':'上海公开目标域 · 校准回放与公开再分析序列；与 PortViz 场景来源独立，不作为现场航行或停工指令。')}`;
        serviceChart?.dispose();serviceChart=window.echarts?.init($('hd-service-chart'),null,{renderer:'svg'});
        const series=source.timeline||[];
        serviceChart?.setOption({animation:false,legend:{top:2,bottom:null,data:['潮位 m','浪高 m'],textStyle:{color:'#9cb9c8',fontSize:10}},grid:{top:40,left:45,right:22,bottom:38},tooltip:{trigger:'axis',confine:true},xAxis:{type:'category',data:series.map(p=>String(p.source_timestamp).replace('T',' ').slice(5,16)),axisLabel:{color:'#7596a7',fontSize:9}},yAxis:{type:'value',name:'m',axisLabel:{color:'#7596a7',fontSize:9},splitLine:{lineStyle:{color:'#253e4b'}}},series:[{name:'潮位 m',type:'line',smooth:true,data:series.map(p=>number(p.tide_m)),itemStyle:{color:'#68dac3'},symbol:'none'},{name:'浪高 m',type:'line',smooth:true,data:series.map(p=>number(p.wave_height_m)),itemStyle:{color:'#81acf5'},symbol:'none'}]});
      }
    } else if(kind==='berth') {
      const p=state.portCall;
      body.innerHTML=`<div class="hd-service-stages">${['到港','引航','拖轮','系泊','装卸','离港'].map((t,i)=>`<div><span>0${i+1}</span><b>${t}</b><small>${p?.boundary?.live_data_verified===true?'源已核验':'现场待接入'}</small></div>`).join('')}</div>${row('靠泊事件源',p?.boundary?.live_data_verified===true?'已验证':'待接入港口')}${row('现场船名 / IMO / ETA','未提供，不生成虚构航次')}${row('支持事件类型',p?.event_contract?.event_types?.length??'未读取')}${note(p?.boundary?.reason||'靠泊服务状态暂不可用。')}`;
      if(state.presentationMode){const s=state.presentation,o=s?.operating;body.innerHTML=o?`${row('场景规模',`${o.berths_total} 个公开泊位 · 2350 m 岸线`)}${row('推演作业中',`${o.working_call_count} 航次 · ${o.qc_busy} 台岸桥`)}${(s.calls||[]).filter(c=>c.phase==='working').map(c=>row(`${c.call_id} · ${c.berth_id}`,`${fmt(c.cargo_teu)} TEU · ${fmt(c.progress*100)}%`)).join('')}${note('航次为明确标识的 DEMO 合成计划，按公开设备规模约束推演；真实船名、IMO、ETA 和引航拖轮服务仍待授权源接入。')}${row('现场事件源',p?.boundary?.live_data_verified===true?'已验证':'待接入')}`:note('公开参数演示源暂不可用，请刷新重试。');}
    } else if(kind==='energy') {
      route='#sbess-section';
      body.innerHTML='<div class="hd-service-empty">读取岸电、储能与冷藏箱负荷…</div>';
      const ids=['shore-01','bess-01','reefer-01'];
      const data=await Promise.allSettled(ids.map(id=>json(energyUrl(id))));
      if(token!==serviceToken||!dialog.open)return;
      body.innerHTML=data.map((result,i)=>{const last=result.status==='fulfilled'&&Array.isArray(result.value)?result.value.at(-1):null;return `<div class="hd-service-energy-row"><span>${safe(state.assets.find(a=>a.id===ids[i])?.label||ids[i])}<small>${last?.measured===true?'现场实测':'校准回放 · 非现场电表'}</small></span><b>${fmt(last?.kW)} <small>kW</small></b><button type="button" data-hd-energy-pick="${ids[i]}">查看曲线 ↗</button></div>`;}).join('')+note(state.presentationMode?'曲线为按工况权重分配的工程机群负荷，不是分项电表，也不代表现场已安装这些能源设施；不推断储能 SOC、接电完成或实际供电状态。':'负荷来自各设备已有遥测契约；未接入现场 BMS 与接电记录前，不推断储能 SOC、接电完成或实际供电状态。');
      body.querySelectorAll('[data-hd-energy-pick]').forEach(button=>button.addEventListener('click',()=>{const select=$('hd-energy-asset');if(![...select.options].some(o=>o.value===button.dataset.hdEnergyPick))return;select.value=button.dataset.hdEnergyPick;loadEnergy();dialog.close();select.scrollIntoView({block:'center',behavior:'smooth'});select.focus({preventScroll:true});}));
    } else {
      route='#strategy-exec-module';const c=state.coordination;
      body.innerHTML=`<div class="hd-service-resource-grid">${(c?.contract?.resource_types||[]).map(r=>`<span>${safe(r.label)}</span>`).join('')}</div>${row('规划步长',c?`${c.contract.fixed_planning.time_step_minutes} 分钟`:'未读取')}${row('计划冻结窗口',c?`${c.contract.fixed_planning.freeze_horizon_minutes} 分钟`:'未读取')}${row('现场协同证据',c?.configured_artifact?.verified===true?'已验证':'待接入现场计划与资源')}${note(c?.boundary?.reason||'资源合同暂不可用，请刷新重试。')}`;
    }
    put('hd-service-source',state.presentationMode&&state.presentation?`历史场景 ${state.presentation.source_timestamp} · 工程演示 / 非现场实况`:kind==='environment'&&sample?`来源时间 ${String(sample.source_timestamp).replace('T',' ').slice(0,19)} UTC`:'公开回放 / 合同登记 · 生产控制未授权');
    $('hd-service-route').href=route;
  }

  function stopTour(){state.tour=false;$('hd-tour').setAttribute('aria-pressed','false');$('hd-tour').textContent='▷ 自动巡航';}
  function setLayer(value) {
    state.layer=value;document.querySelectorAll('[data-hd-layer]').forEach(button=>{const selected=button.dataset.hdLayer===value;button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));});state.scene?.layer(value);
  }

  async function createScene(layout) {
    // Browsers cache failed module imports. A new attempt must use a new URL;
    // a successfully loaded engine is reused for context recovery.
    if(!sceneLibrary) {
      const attempt=++libraryAttempt;
      try {sceneLibrary=await import(`/static/vendor/three/three.module.min.js?v=160.1${attempt>1?`&retry=${attempt}`:''}`);}
      catch(_) {throw new Error('三维引擎资源暂不可用，请重试');}
    }
    const T = sceneLibrary;
    const host = $('hd-scene-host');
    const renderer = new T.WebGLRenderer({antialias:true,alpha:false,powerPreference:'low-power'});
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1,1.65)); renderer.setClearColor(0x0c1d2a,1);
    renderer.outputColorSpace = T.SRGBColorSpace; renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.3;
    renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;renderer.shadowMap.autoUpdate=false;
    renderer.domElement.setAttribute('aria-hidden','true'); host.prepend(renderer.domElement);
    const scene = new T.Scene(); scene.fog = new T.FogExp2(0x0c1d2a, .0012);
    const camera = new T.OrthographicCamera(-140,140,85,-85,.1,900);
    const ambient=new T.HemisphereLight(0xc8e5f1,0x243c42,2.1);scene.add(ambient);
    const sun = new T.DirectionalLight(0xffeed3,3); sun.position.set(-70,150,80); scene.add(sun);
    sun.castShadow=true;sun.shadow.mapSize.set(1024,1024);Object.assign(sun.shadow.camera,{left:-135,right:135,top:110,bottom:-110,near:1,far:400});sun.shadow.bias=-.0005;sun.shadow.normalBias=.04;
    const rim = new T.DirectionalLight(0x55bdcf,2); rim.position.set(70,40,-100); scene.add(rim);
    const worldW = number(layout.world?.W) || 1600, worldH = number(layout.world?.H) || 900;
    const scale = 160 / worldW;
    const x = value => (value - worldW/2)*scale, z = value => (value - worldH/2)*scale;
    const base = new T.Group(), yard = new T.Group(), quay = new T.Group(), transport = new T.Group(), facilities=new T.Group(), routeLines=new T.Group();
    scene.add(base,yard,quay,transport,facilities,routeLines);
    const materials = new Map();
    const mat = color => { if (!materials.has(color)) materials.set(color,new T.MeshStandardMaterial({color,roughness:.72,metalness:.15})); return materials.get(color); };
    const box = (group,w,h,d,px,py,pz,color) => { const mesh = new T.Mesh(new T.BoxGeometry(w,h,d),mat(color)); mesh.position.set(px,py,pz);mesh.castShadow=h>2;mesh.receiveShadow=true; group.add(mesh); return mesh; };
    const lineMats=new Map();
    const line = (group,points,color,opacity=1) => {const key=`${color}/${opacity}`;if(!lineMats.has(key))lineMats.set(key,new T.LineBasicMaterial({color,transparent:opacity<1,opacity}));const mesh = new T.Line(new T.BufferGeometry().setFromPoints(points.map(p => new T.Vector3(...p))),lineMats.get(key)); group.add(mesh); return mesh; };
    const beam = (group,a,b,width,color) => { const mid = new T.Vector3(...a).add(new T.Vector3(...b)).multiplyScalar(.5); const vector = new T.Vector3(...b).sub(new T.Vector3(...a)); const mesh = box(group,width,vector.length(),width,mid.x,mid.y,mid.z,color); mesh.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),vector.normalize()); return mesh; };
    const textures=[];
    function textPlate(group,text,px,py,pz,width=7,color='#d7ddd4',flat=false) {
      const canvas=document.createElement('canvas');canvas.width=256;canvas.height=64;const ctx=canvas.getContext('2d');ctx.font='600 35px sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,128,32);
      const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;textures.push(texture);
      const plate=new T.Mesh(new T.PlaneGeometry(width,width/4),new T.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,side:T.DoubleSide}));plate.position.set(px,py,pz);if(flat)plate.rotation.x=-Math.PI/2;group.add(plate);return plate;
    }
    function containerMaterial(color) {
      const key=`container-${color}`;if(materials.has(key))return materials.get(key);
      const canvas=document.createElement('canvas');canvas.width=128;canvas.height=64;const ctx=canvas.getContext('2d');ctx.fillStyle=`#${color.toString(16).padStart(6,'0')}`;ctx.fillRect(0,0,128,64);ctx.fillStyle='#00000028';for(let i=4;i<128;i+=9)ctx.fillRect(i,0,2,64);ctx.strokeStyle='#ffffff35';ctx.strokeRect(2,2,124,60);ctx.fillStyle='#ffffffa0';ctx.font='bold 11px sans-serif';ctx.fillText('TWINPORT',35,35);
      const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;texture.anisotropy=renderer.capabilities.getMaxAnisotropy();textures.push(texture);
      const material=new T.MeshStandardMaterial({map:texture,roughness:.76,metalness:.12});materials.set(key,material);return material;
    }
    const glowMaterials=[];
    function glow(group,w,h,d,px,py,pz,color=0x9de9e1) {const mesh=box(group,w,h,d,px,py,pz,color),key=`glow-${color}`;if(!materials.has(key)){const material=new T.MeshBasicMaterial({color});materials.set(key,material);glowMaterials.push(material);}mesh.material=materials.get(key);return mesh;}
    const grid = new T.GridHelper(400,50,0x284654,0x183544); grid.position.y=-2.2; base.add(grid);
    const berth = layout.berth || {x:120,y:60,w:1360,h:24}, quayZ = z(berth.y+berth.h);
    const minZ = Math.min(quayZ,z(0)), maxZ = z(worldH);
    box(base,176,3,maxZ-quayZ,0,-1,(maxZ+quayZ)/2,0x233b47);
    box(base,174,.3,4,0,.7,quayZ+2,0x4b5c5c);
    box(base,174,.35,.4,0,.9,quayZ+.2,0x95b0a8);
    // Quiet water with dashed navigation markings and geometric wave highlights.
    const waterGeometry=new T.PlaneGeometry(280,72,72,22);waterGeometry.rotateX(-Math.PI/2);
    const water=new T.Mesh(waterGeometry,new T.MeshPhongMaterial({color:0x174d63,specular:0x163e50,shininess:80,transparent:true,opacity:.96}));water.position.set(0,-1.9,quayZ-37);base.add(water);
    const waterBase=waterGeometry.attributes.position.array.slice();
    for (let i=0;i<25;i++) { const px=-116+i*9; const zz=quayZ-8-(i%5)*8; line(base,[[px,-1.9,zz],[px+5,-1.9,zz]],0x36717c,.27); }
    line(base,[[-125,-1.8,quayZ-52],[120,-1.8,quayZ-52]],0x3b7a8b,.3);
    // Perimeter and road markings.
    line(base,[[-88,.6,quayZ],[-88,.6,maxZ],[88,.6,maxZ],[88,.6,quayZ]],0x628982,.6);
    (layout.lanes || []).forEach(lane => {
      if (lane.length < 2) return;
      const a = lane[0], b = lane.at(-1);
      const road=box(base,Math.hypot(b.x-a.x,b.y-a.y)*scale,.05,3.7,(x(a.x)+x(b.x))/2,.78,(z(a.y)+z(b.y))/2,0x1d2a31);road.rotation.y=-Math.atan2(b.y-a.y,b.x-a.x);
      line(routeLines,[[x(a.x),1.05,z(a.y)],[x(b.x),1.05,z(b.y)]],0x5ebfba,.28);
      const length = Math.hypot(b.x-a.x,b.y-a.y)*scale;
      for(let at=0;at<length;at+=5) { const t = at/length, end = Math.min(1,(at+2)/length); line(base,[[x(a.x+(b.x-a.x)*t),.9,z(a.y+(b.y-a.y)*t)],[x(a.x+(b.x-a.x)*end),.9,z(a.y+(b.y-a.y)*end)]],0xc1c1a3,.6); }
    });
    const pickables = [], qcModels = [], ycModels = [], agvModels = [], truckModels = [], vesselModels = [];
    const largeFleet=layout.meta?.source_mode==='public_parameter_calibrated_demo';
    let ycStatusInstances=null,agvInstances=null;
    // Container modules are instanced; 3D remains smooth on integrated GPUs.
    const blocks = Array.isArray(layout.yards) ? layout.yards : [];
    const colors = [0x3a9a9a,0x466d99,0xc38549,0x9a6254,0x739791];
    const boxes = colors.map(color => new T.InstancedMesh(new T.BoxGeometry(3.2,1.65,1.35),containerMaterial(color),Math.max(1,blocks.length*30)));
    const counts = colors.map(() => 0), dummy = new T.Object3D();
    blocks.forEach((block,index) => {
      box(yard,block.w*scale+.8,.18,block.h*scale+.8,x(block.x+block.w/2),.7,z(block.y+block.h/2),0x1a303c);
      textPlate(yard,`Y-${String(index+1).padStart(2,'0')}`,x(block.x+block.w/2),.96,z(block.y)-1.2,5,'#bbc5a7',true);
      for(let col=0;col<3;col++) for(let row=0;row<3;row++) for(let level=0;level<2+(index%3===0?1:0);level++) {
        const colorIndex = (index+col+row)%colors.length;
        dummy.position.set(x(block.x)+2+col*3.7,1.6+level*1.72,z(block.y)+.9+row*1.6); dummy.updateMatrix(); boxes[colorIndex].setMatrixAt(counts[colorIndex]++,dummy.matrix);
      }
      const hit = box(yard,block.w*scale,7,block.h*scale,x(block.x+block.w/2),3.5,z(block.y+block.h/2),0x000000);
      hit.material = new T.MeshBasicMaterial({visible:false}); hit.userData={focus:'yard',index}; pickables.push(hit);
    });
    boxes.forEach((mesh,i) => {mesh.count=counts[i];mesh.castShadow=true;mesh.receiveShadow=true;mesh.instanceMatrix.needsUpdate=true;yard.add(mesh);});
    (layout.qcs || []).forEach((pos,index) => {
      const model = new T.Group(); model.position.set(x(pos.x),1,z(pos.y)); quay.add(model);
      const structural = 0xcfb478, dark = 0x526472;
      [-2.1,2.1].forEach(px => {
        [-2.4,2.4].forEach(pz => { beam(model,[px,0,pz],[px,17,pz*.58],.5,structural); box(model,1.6,.8,1.2,px,.2,pz,dark); });
        beam(model,[px,17,1.5],[px,24,-.5],.38,structural);
        beam(model,[px,24,-.5],[px,18,-15],.15,0xa6b9b3);
        beam(model,[px,24,-.5],[px,18,8],.15,0xa6b9b3);
        box(model,.55,.8,25,px,18,-3.5,structural);
      });
      box(model,5,.7,1,0,17,-1,structural);box(model,5,.5,1,0,18,5,structural);
      const trolley = box(model,4,.7,2,0,17,-5,0x8b9e9d);
      const cable = box(model,.08,7,.08,0,13,-5,0x849da4), spreader=box(model,3.2,.4,1.7,0,9.4,-5,0x6eb6ac);
      box(model,2,1.7,2,2.8,15.5,2,0x467985);
      textPlate(model,`QC-${String(index+1).padStart(2,'0')}`,0,16.5,2.6,3.6,'#fff1bd');
      for(let r=0;r<(largeFleet?0:5);r++)beam(model,[-2.1,r*3,2.4],[-2.1,r*3+3,1.5],.09,0xa5aaa0);
      const status = box(model,1,.7,1,0,19.1,5,0x66cbbb);
      glow(model,.4,.25,.4,2.1,24.3,-.5,0xedb16d);
      const hit = box(model,7,24,24,0,11,-3.5,0x000000);hit.material=new T.MeshBasicMaterial({visible:false});hit.userData={focus:'quay',index};pickables.push(hit);
      qcModels.push({model,trolley,cable,spreader,status,index});
    });
    // Grounded RTG frames, with contrasting tops for legible yard structure.
    if(largeFleet){
      const parts=[[-3.7,2.5,-1.6,.35,5,.35],[-3.7,2.5,1.6,.35,5,.35],[3.7,2.5,-1.6,.35,5,.35],[3.7,2.5,1.6,.35,5,.35],[-3.7,5,0,.5,.5,3.8],[3.7,5,0,.5,.5,3.8],[0,5,0,8,.6,.7]];
      const fleet=new T.InstancedMesh(new T.BoxGeometry(1,1,1),mat(0x99a4bb),(layout.ycs||[]).length*parts.length);
      const stamp=new T.Object3D();let n=0;
      (layout.ycs||[]).forEach(pos=>parts.forEach(([px,py,pz,w,h,d])=>{stamp.position.set(x(pos.x)+px,.7+py,z(pos.y)+pz);stamp.scale.set(w,h,d);stamp.updateMatrix();fleet.setMatrixAt(n++,stamp.matrix);}));
      fleet.instanceMatrix.needsUpdate=true;yard.add(fleet);
      ycStatusInstances=new T.InstancedMesh(new T.BoxGeometry(1,.4,1),mat(0xffffff),(layout.ycs||[]).length);
      (layout.ycs||[]).forEach((pos,i)=>{stamp.position.set(x(pos.x),6.3,z(pos.y));stamp.scale.set(1,1,1);stamp.updateMatrix();ycStatusInstances.setMatrixAt(i,stamp.matrix);ycStatusInstances.setColorAt(i,new T.Color(0x647786));});
      ycStatusInstances.instanceMatrix.needsUpdate=true;yard.add(ycStatusInstances);
    }
    (largeFleet?[]:(layout.ycs || [])).forEach((pos,index) => {
      const group = new T.Group(); group.position.set(x(pos.x),.7,z(pos.y));yard.add(group);
      [-3.7,3.7].forEach(px => {[-1.6,1.6].forEach(pz => {box(group,.35,5,.35,px,2.5,pz,0x799096);box(group,.7,.55,.6,px,.2,pz,0x182832);});box(group,.5,.5,3.8,px,5,0,0xa3a0cb);});
      box(group,8,.6,.7,0,5,0,0xa3a0cb);const status=box(group,1,.4,1,0,5.6,0,0x647786);
      textPlate(group,`YC-${String(index+1).padStart(2,'0')}`,0,4.6,1.9,3.2,'#cddde4');
      ycModels.push({group,status,index});
    });
    // A low administration/service district keeps the land edge readable.
    for(let i=0;i<3;i++) {
      const wx=-57+i*25; const wz=maxZ-5;
      box(base,16,4,7,wx,2,wz,0x476373);box(base,16.5,.35,7.4,wx,4.1,wz,0x718d98);
      for(let k=0;k<6;k++) box(base,1.1,.6,.08,wx-6+k*2.3,2.5,wz+3.55,0x8acbd0);
    }
    for(let i=0;i<9;i++) { const px=-78+i*19; beam(base,[px,.5,maxZ-10],[px,8,maxZ-10],.14,0x7b9395);glow(base,2,.12,.5,px,8,maxZ-10,0xffe8b0); }
    // Quayside fittings and numbered stand areas are geometric illustrations.
    for(let i=0;i<22;i++) {const px=-80+i*7.6;box(base,.9,1.6,.65,px,-.2,quayZ-.35,0x14222c);box(base,.6,.45,.8,px,1.15,quayZ+1.1,0xc3b291);}
    for(let i=0;i<(layout.berth?.count||4);i++)textPlate(base,`B${String(i+1).padStart(2,'0')}`, x((layout.berth?.x||120)+(i+.5)*(layout.berth?.w||1360)/(layout.berth?.count||4)),.95,quayZ+6,7,'#e0d4aa',true);
    // Service district: substations, BESS, reefer connections, gate and rail.
    const facilityHits=[];
    function facilityHit(key,px,pz,w,d){const hit=box(facilities,w,9,d,px,4,pz,0);hit.material=new T.MeshBasicMaterial({visible:false});hit.userData.focus=key;facilityHits.push(hit);pickables.push(hit);}
    box(facilities,22,.25,19,70,.85,-6,0x30494f);
    for(let i=0;i<3;i++) {
      box(facilities,4.5,3.7,7,64+i*5.8,2.7,-5,0x91b1ae);
      box(facilities,4.5,.25,7.1,64+i*5.8,4.65,-5,0xb5c5bb);
      for(let vent=0;vent<4;vent++)box(facilities,2.8,.16,.08,64+i*5.8,2+vent*.5,-1.42,0x42656a);
      glow(facilities,.18,.18,.1,65+i*5.8,3.7,-1.35,0x6ee4ba);
    }
    for(let i=0;i<4;i++) {box(facilities,2.1,1.8,2,62+i*5,1.8,-13,0x718b91);glow(facilities,.15,.15,.1,62+i*5,2.2,-11.95);}
    const solar=box(facilities,20,.3,7,70,6,-5,0x24485d);solar.rotation.z=-.1;
    for(let i=0;i<6;i++)line(facilities,[[60+i*3.3,6.2,-8.4],[60+i*3.3,6.2,-1.6]],0x82a5ae,.55);
    [-9,-1].forEach(pz=>[-9,9].forEach(px=>beam(facilities,[70+px,1,pz],[70+px,6,pz],.15,0x8da2a3)));
    textPlate(facilities,'SHORE / BESS',70,.99,5,17,'#a4d2c8',true);facilityHit('energy',70,-6,23,20);
    for(let i=0;i<4;i++) {
      const px=-48+i*27;box(facilities,1.5,2.5,1.2,px,2,quayZ+3,0xb2c5ad);
      line(facilities,[[px,1.1,quayZ+3],[px,1.1,quayZ-4],[px+2,2.2,quayZ-8]],0x68c6ba,.8);
    }
    box(facilities,23,.4,10,65,1,34,0x35454a);
    box(facilities,24,.65,8,65,7,34,0x62858b);
    for(let i=0;i<5;i++){const px=54+i*5.5;beam(facilities,[px,1,31],[px,7,31],.27,0xa6b9b4);box(facilities,.35,2,.3,px,2,38,0xc6ac77);glow(facilities,.45,.35,.2,px,6.2,38.1,0x68d5b0);box(facilities,3.8,.16,.2,px+2,2.8,38,0xe7d7ae);}
    textPlate(facilities,'GATE',65,7.1,38.1,10,'#e5ead8');facilityHit('gate',65,34,25,12);
    // Railway is a labelled service model, not an invented operating train.
    [-.6,.6].forEach(offset=>line(facilities,[[10,.98,39+offset],[83,.98,39+offset]],0x9aa6a1));
    for(let px=10;px<83;px+=2)box(facilities,.25,.1,2.7,px,.9,39,0x6b6860);
    for(let i=0;i<4;i++){box(facilities,7,.5,2.6,14+i*8.5,1.4,39,0x4c6469);const cargo=box(facilities,6.5,2.7,2.3,14+i*8.5,3.1,39,colors[i%colors.length]);cargo.material=containerMaterial(colors[i%colors.length]);}
    // Safety perimeter and ground lanes make the scale easier to read.
    for(let i=0;i<45;i++){const px=-86+i*3.9;beam(base,[px,.8,maxZ],[px,2,maxZ],.08,0x637a79);}
    line(base,[[-86,1.8,maxZ],[86,1.8,maxZ]],0x758c8a,.6);
    // Collapse repeated static parts into a handful of draws without simplifying
    // their geometry. Dynamic crane mechanisms and picking proxies stay separate.
    if(largeFleet){
      const dynamic=new Set(qcModels.flatMap(q=>[q.trolley,q.cable,q.spreader,q.status]));
      for(const root of [base,yard,quay,facilities,routeLines]){
        root.updateMatrixWorld(true);const inverse=root.matrixWorld.clone().invert(),groups=new Map(),lines=new Map();
        root.traverse(object=>{
          if(object.isInstancedMesh||dynamic.has(object)||object.material?.visible===false)return;
          if(object.isMesh&&object.geometry?.type==='BoxGeometry'){
            const key=object.material.uuid+'/'+object.castShadow;let list=groups.get(key);if(!list){list=[];groups.set(key,list);}list.push(object);
          }else if(object.isLine&&!object.isLineSegments){let list=lines.get(object.material.uuid);if(!list){list=[];lines.set(object.material.uuid,list);}list.push(object);}
        });
        for(const list of groups.values()){
          if(list.length<2)continue;
          const batch=new T.InstancedMesh(new T.BoxGeometry(1,1,1),list[0].material,list.length);batch.castShadow=list[0].castShadow;batch.receiveShadow=true;
          list.forEach((object,i)=>{const p=object.geometry.parameters,matrix=inverse.clone().multiply(object.matrixWorld).multiply(new T.Matrix4().makeScale(p.width,p.height,p.depth));batch.setMatrixAt(i,matrix);object.removeFromParent();object.geometry.dispose();});
          batch.instanceMatrix.needsUpdate=true;root.add(batch);
        }
        for(const list of lines.values()){
          if(list.length<2)continue;const vertices=[];
          for(const object of list){const positions=object.geometry.attributes.position,matrix=inverse.clone().multiply(object.matrixWorld),p=new T.Vector3();for(let i=1;i<positions.count;i++)for(const at of[i-1,i]){p.fromBufferAttribute(positions,at).applyMatrix4(matrix);vertices.push(p.x,p.y,p.z);}object.removeFromParent();object.geometry.dispose();}
          const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(vertices,3));root.add(new T.LineSegments(geometry,list[0].material));
        }
      }
    }
    const makeVehicle = truck => {
      const group = new T.Group();
      box(group,truck?3.5:2.5,.55,1.4,0,1,0,truck?0x8a9aa2:0x75bdb2);
      box(group,.65,.65,1.15,1.3,1.4,0,0xc1c7b8);
      if(truck)box(group,2.4,.8,1.3,-.25,1.55,0,0x597b9b);
      [-.9,.9].forEach(px => [-.65,.65].forEach(pz => box(group,.45,.45,.3,px,.65,pz,0x172631)));
      group.userData.target = new T.Vector3(); transport.add(group); return group;
    };
    const ensureVehicles = frame => {
      if(largeFleet&&!agvInstances){
        const templates=[[0,1,0,2.5,.55,1.4,0x75bdb2],[1.3,1.4,0,.65,.65,1.15,0xc1c7b8],[-.9,.65,-.65,.45,.45,.3,0x172631],[-.9,.65,.65,.45,.45,.3,0x172631],[.9,.65,-.65,.45,.45,.3,0x172631],[.9,.65,.65,.45,.45,.3,0x172631]];
        agvInstances=templates.map(([px,py,pz,w,h,d,color])=>{const mesh=new T.InstancedMesh(new T.BoxGeometry(w,h,d),mat(color),512);mesh.frustumCulled=false;transport.add(mesh);return{mesh,offset:new T.Vector3(px,py,pz)};});
      }
      while(agvModels.length<frame.agv.length){if(largeFleet){const model=new T.Object3D();model.userData.target=new T.Vector3();agvModels.push(model);}else agvModels.push(makeVehicle(false));}
      if(agvInstances)agvInstances.forEach(({mesh})=>{mesh.count=frame.agv.length;});
      while(truckModels.length<frame.tr.length) truckModels.push(makeVehicle(true));
      agvModels.forEach((model,i) => {model.visible=i<frame.agv.length;});truckModels.forEach((model,i) => {model.visible=i<frame.tr.length;});
      while(vesselModels.length<(frame.vessels || []).length) {
        const group = new T.Group(); quay.add(group);
        const hull = new T.Shape();hull.moveTo(-28,-4.8);hull.lineTo(23,-4.8);hull.lineTo(31,0);hull.lineTo(23,4.8);hull.lineTo(-28,4.8);hull.closePath();
        const hullMesh = new T.Mesh(new T.ExtrudeGeometry(hull,{depth:3.5,bevelEnabled:true,bevelThickness:.5,bevelSize:.6,bevelSegments:1,steps:1}),mat(0x275266));hullMesh.castShadow=true;hullMesh.receiveShadow=true;
        hullMesh.rotation.x=-Math.PI/2;hullMesh.position.y=-1;group.add(hullMesh);
        box(group,48,.3,8,-2,3.1,0,0x758c8a);
        if(largeFleet){
          const cargoMeshes=colors.map(c=>new T.InstancedMesh(new T.BoxGeometry(3.1,1.5,2),containerMaterial(c),60)),counts=colors.map(()=>0),stamp=new T.Object3D();
          for(let col=0;col<10;col++)for(let row=0;row<3;row++)for(let level=0;level<2;level++){const ci=(col+row)%colors.length;stamp.position.set(-18+col*3.6,4.1+level*1.6,-2.5+row*2.3);stamp.updateMatrix();cargoMeshes[ci].setMatrixAt(counts[ci]++,stamp.matrix);}
          cargoMeshes.forEach((mesh,i)=>{mesh.count=counts[i];mesh.instanceMatrix.needsUpdate=true;group.add(mesh);});
        }else for(let col=0;col<10;col++)for(let row=0;row<3;row++)for(let level=0;level<2;level++){const cargo=box(group,3.1,1.5,2,-18+col*3.6,4.1+level*1.6,-2.5+row*2.3,colors[(col+row)%colors.length]);cargo.material=containerMaterial(colors[(col+row)%colors.length]);}
        box(group,5,6.4,8,-24,5.5,0,0xb6c3bd);box(group,5.3,.8,8.2,-24,9,0,0x5c9ca4);
        box(group,.8,3,1,-24,10.5,0,0xc1b699);
        for(let i=0;i<8;i++)glow(group,.38,.36,.1,-26.2+i*.64,8.6,4.17,0xb0e5e0);
        [-4.3,4.3].forEach(pz=>{line(group,[[-27,3.8,pz],[23,3.8,pz]],0xc9d3c3);for(let i=0;i<(largeFleet?0:17);i++)beam(group,[-27+i*3,3.1,pz],[-27+i*3,3.8,pz],.045,0xa8b8b1);});
        line(group,[[-27,.1,4.95],[23,.1,4.95],[30,.1,0],[23,.1,-4.95],[-27,.1,-4.95]],0xb77656);
        textPlate(group,`VIS-V${String(vesselModels.length+1).padStart(3,'0')}`,1,1.6,5,14,'#d7e4de');
        const hit=box(group,60,7,10,0,3,0,0);hit.material=new T.MeshBasicMaterial({visible:false});hit.userData={focus:'vessel',index:vesselModels.length};pickables.push(hit);
        vesselModels.push(group);
      }
      vesselModels.forEach((model,i) => {model.visible=i<(frame.vessels || []).length;});
    };
    let initializedPositions = false;
    function update(frame) {
      ensureVehicles(frame);
      qcModels.forEach(({model,trolley,cable,spreader,status,index}) => {
        const data = frame.qc[index]; if(!data)return;
        const p = -12+Math.max(0,Math.min(1,number(data.trolley) || 0))*18;
        model.userData.trolleyTarget=p;model.userData.hoistTarget=8+Math.sin((number(data.trolley)||0)*Math.PI)*5;
        status.material=mat(data.busy===true?0x75d9b7:0x536678);
      });
      ycModels.forEach(({status,index}) => {status.material=mat(frame.yc[index]?.busy===true?0x97c9cb:0x647786);});
      if(ycStatusInstances){frame.yc.forEach((data,i)=>{if(i<ycStatusInstances.count)ycStatusInstances.setColorAt(i,new T.Color(data.busy===true?0x97c9cb:0x647786));});ycStatusInstances.instanceColor.needsUpdate=true;}
      frame.agv.forEach((data,index) => {
        const lane = layout.lanes?.[data.lane]; if(!lane?.length)return;
        // Source distance s is a percentage of polyline length, including bends.
        const segments=lane.slice(1).map((p,i) => Math.hypot(p.x-lane[i].x,p.y-lane[i].y));
        let distance=segments.reduce((a,b)=>a+b,0)*Math.max(0,Math.min(100,number(data.s)||0))/100;
        let segment=0;while(segment<segments.length-1 && distance>segments[segment]){distance-=segments[segment];segment++;}
        const a=lane[segment], b=lane[segment+1], ratio=segments[segment]?distance/segments[segment]:0;
        const model=agvModels[index];model.userData.target.set(x(a.x+(b.x-a.x)*ratio),0,z(a.y+(b.y-a.y)*ratio));model.rotation.y=-Math.atan2(b.y-a.y,b.x-a.x);
        if(!initializedPositions || model.position.distanceTo(model.userData.target)>30)model.position.copy(model.userData.target);
      });
      frame.tr.forEach((data,index) => {const model=truckModels[index];model.userData.target.set(x(data.x),0,z(data.y));if(!initializedPositions || model.position.distanceTo(model.userData.target)>30)model.position.copy(model.userData.target);});
      (frame.vessels || []).forEach((data,index) => {const model=vesselModels[index];const length=number(data.len);model.scale.x=length===null?1:Math.max(.25,length*scale/60);if(largeFleet){const count=layout.berth?.count||7;model.scale.set((length||250)*(layout.berth.w*scale/2350)/60,.65,.6);model.position.set(x(layout.berth.x+(data.berth+.5)*layout.berth.w/count),-.5,quayZ-5.5);}else model.position.set(-12+index*50,-.5,quayZ-11-index*13);});
      initializedPositions=true;
      renderer.shadowMap.needsUpdate=true;
    }

    let yaw=2.65, elevation=.77, zoom=1.02, targetYaw=yaw,targetElevation=elevation,targetZoom=zoom;
    const target = new T.Vector3(0,0,-6), aim = target.clone();
    const anchors={quay:new T.Vector3(-40,18,quayZ-1),yard:new T.Vector3(24,8,z(300)),transport:new T.Vector3(16,2,z(670)),energy:new T.Vector3(70,7,-6),gate:new T.Vector3(65,7,34),vessel:new T.Vector3(-10,5,quayZ-13)};
    const point = new T.Vector3(); let raf=0,last=0,disposed=false,contextLost=false,drag;
    let focusKey='quay';
    function resize() {
      const width=host.clientWidth,height=host.clientHeight;if(!width||!height)return;
      renderer.setSize(width,height,false);camera.left=-81*width/height;camera.right=81*width/height;camera.top=81;camera.bottom=-81;camera.updateProjectionMatrix();energyChart?.resize();serviceChart?.resize();
      wake();
    }
    let animationTime=0;
    function draw(time) {
      raf=0;if(disposed || contextLost || !active() || ($('hd-service-dialog').open||document.querySelector('.ppd-dialog[open]')))return;
      if(time-last<32 && !motionQuery.matches){raf=requestAnimationFrame(draw);return;}
      const delta=last?Math.min(.1,(time-last)/1000):.033;last=time;
      if(!state.paused&&!motionQuery.matches)animationTime+=delta;
      yaw+=(targetYaw-yaw)*.12;elevation+=(targetElevation-elevation)*.12;zoom+=(targetZoom-zoom)*.12;target.lerp(aim,.12);
      camera.position.set(target.x+Math.sin(yaw)*220*Math.cos(elevation),target.y+220*Math.sin(elevation),target.z+Math.cos(yaw)*220*Math.cos(elevation));camera.zoom=zoom*Math.min(1,(host.clientWidth/host.clientHeight)/1.25);camera.lookAt(target);camera.updateProjectionMatrix();
      if(!state.paused) [...agvModels,...truckModels].forEach(model => {if(model.visible)model.position.lerp(model.userData.target,motionQuery.matches?1:.12);});
      if(agvInstances){const stamp=new T.Object3D();for(const {mesh,offset}of agvInstances){for(let i=0;i<agvModels.length;i++){const vehicle=agvModels[i];stamp.position.copy(offset).applyQuaternion(vehicle.quaternion).add(vehicle.position);stamp.quaternion.copy(vehicle.quaternion);stamp.updateMatrix();mesh.setMatrixAt(i,stamp.matrix);}mesh.instanceMatrix.needsUpdate=true;}}
      qcModels.forEach(({model,trolley,cable,spreader})=>{
        const p=model.userData.trolleyTarget??-5,h=model.userData.hoistTarget??9.4;
        trolley.position.z+=(p-trolley.position.z)*(state.paused?1:.1);cable.position.z=spreader.position.z=trolley.position.z;
        spreader.position.y+=(h-spreader.position.y)*(state.paused?1:.1);cable.scale.y=Math.max(.15,(17-spreader.position.y)/7);cable.position.y=(17+spreader.position.y)/2;
      });
      const vertices=waterGeometry.attributes.position;
      for(let i=0;i<vertices.count;i++){const px=waterBase[i*3],pz=waterBase[i*3+2];vertices.setY(i,Math.sin(px*.28+animationTime*.9)*.08+Math.cos(pz*.49-animationTime*.7)*.06);}
      vertices.needsUpdate=true;waterGeometry.computeVertexNormals();
      water.position.y=-1.9;
      const submitStarted=performance.now();renderer.render(scene,camera);
      if(!host.dataset.readyAtMs)host.dataset.readyAtMs=performance.now().toFixed(1);
      if(!host.dataset.perfAt||time-Number(host.dataset.perfAt)>1500){host.dataset.perfAt=String(time);host.dataset.renderCalls=String(renderer.info.render.calls);host.dataset.renderSubmitMs=(performance.now()-submitStarted).toFixed(2);}
      Object.entries(anchors).forEach(([key,anchor]) => {
        point.copy(anchor).project(camera);const label=$(`hd-label-${key}`);if(!label)return;
        const left=`${((point.x*.5+.5)*host.clientWidth).toFixed(1)}px`,top=`${((-point.y*.5+.5)*host.clientHeight).toFixed(1)}px`;
        if(label.style.left!==left)label.style.left=left;if(label.style.top!==top)label.style.top=top;
        const shown=state.layer==='services'?['energy','gate','quay'].includes(key):state.layer==='transport'?key==='transport':['quay','yard','transport'].includes(key);
        const visible=state.labels&&shown&&point.z<1&&Math.abs(point.x)<.86&&Math.abs(point.y)<.77;
        const opacity=visible?'1':'0',visibility=visible?'visible':'hidden';
        if(label.style.opacity!==opacity)label.style.opacity=opacity;if(label.style.visibility!==visibility)label.style.visibility=visibility;
      });
      const moving=Math.abs(yaw-targetYaw)>.0005||Math.abs(elevation-targetElevation)>.0005||Math.abs(zoom-targetZoom)>.0005||target.distanceTo(aim)>.01;
      if((!motionQuery.matches && !state.paused) || moving) raf=requestAnimationFrame(draw);
    }
    function wake(){if(!raf && active() && !disposed && !contextLost)raf=requestAnimationFrame(draw);}
    const reset = () => {stopTour();if(state.layer!=='overview')setLayer('overview');targetYaw=2.65;targetElevation=.77;targetZoom=1.02;aim.set(0,0,-6);focusKey='quay';$('hd-camera').value='overview';$('hd-scene-inspector').hidden=true;wake();};
    const focus = key => {if(state.layer!=='overview')setLayer(['energy','gate'].includes(key)?'services':key==='transport'?'transport':'overview');focusKey=key;aim.copy(anchors[key] || anchors.quay);aim.y=0;targetZoom=['energy','gate'].includes(key)?2.1:key==='vessel'?1.8:1.4;targetYaw=key==='quay'||key==='vessel'?2.95:key==='energy'?2.1:2.65;targetElevation=key==='quay'?.6:.77;if([...$('hd-camera').options].some(o=>o.value===key))$('hd-camera').value=key;wake();};
    const lightPools=new T.Group();scene.add(lightPools);
    for(let i=0;i<9;i++){const pool=new T.Mesh(new T.CircleGeometry(7,24),new T.MeshBasicMaterial({color:0xffdf8b,transparent:true,opacity:.1,depthWrite:false}));pool.rotation.x=-Math.PI/2;pool.position.set(-78+i*19,.95,maxZ-10);lightPools.add(pool);}
    const energyLight=new T.PointLight(0x6accca,70,35,1.5);energyLight.position.set(70,6,-6);scene.add(energyLight);
    const gateLight=new T.PointLight(0xffd59a,65,30,1.5);gateLight.position.set(65,10,34);scene.add(gateLight);
    const quayLights=[-50,0,50].map(px=>{const light=new T.PointLight(0xb9dcdb,180,90,1.5);light.position.set(px,22,quayZ+6);scene.add(light);return light;});
    function theme(value) {
      const night=value==='night';ambient.intensity=night?1:2.1;sun.intensity=night?.35:3;rim.intensity=night?1.5:2;
      renderer.setClearColor(night?0x07131f:0x102632,1);scene.fog.color.set(night?0x07131f:0x102632);renderer.toneMappingExposure=night?1.5:1.25;
      water.material.color.set(night?0x0c293e:0x174d63);lightPools.visible=night;energyLight.visible=gateLight.visible=night;renderer.shadowMap.needsUpdate=true;
      quayLights.forEach(light=>{light.visible=night;});
      host.dataset.theme=value;wake();
    }
    const raycaster = new T.Raycaster(), pointer = new T.Vector2();
    const inputController=new AbortController(),inputOptions={signal:inputController.signal};
    host.addEventListener('pointerdown',event => {if(event.target!==renderer.domElement || event.button!==0)return;stopTour();drag={x:event.clientX,y:event.clientY,yaw:targetYaw,elevation:targetElevation,moved:false};host.setPointerCapture(event.pointerId);},inputOptions);
    host.addEventListener('pointermove',event => {if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;targetYaw=drag.yaw-dx*.006;targetElevation=Math.max(.32,Math.min(1.35,drag.elevation+dy*.004));wake();},inputOptions);
    host.addEventListener('pointerup',event => {
      if(!drag)return;const clicked=!drag.moved;drag=null;if(host.hasPointerCapture(event.pointerId))host.releasePointerCapture(event.pointerId);
      if(clicked){const rect=host.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);const hit=raycaster.intersectObjects(pickables.filter(mesh=>{let node=mesh;while(node){if(!node.visible)return false;node=node.parent;}return true;}),false)[0];if(hit)inspect(hit.object.userData.focus,true,hit.object.userData.index??null);}
    },inputOptions);
    host.addEventListener('pointercancel',()=>{drag=null;},inputOptions);
    host.addEventListener('wheel',event => {event.preventDefault();targetZoom=Math.max(.65,Math.min(2.5,targetZoom*Math.exp(-event.deltaY*.001)));wake();},{passive:false,...inputOptions});
    host.addEventListener('keydown',event => {
      if(event.target!==host)return;const keys=['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','0'];if(!keys.includes(event.key))return;event.preventDefault();
      if(event.key==='ArrowLeft')targetYaw-=.12;if(event.key==='ArrowRight')targetYaw+=.12;if(event.key==='ArrowUp')targetElevation=Math.min(1.35,targetElevation+.07);if(event.key==='ArrowDown')targetElevation=Math.max(.32,targetElevation-.07);if(['+','='].includes(event.key))targetZoom=Math.min(2.5,targetZoom+.1);if(event.key==='-')targetZoom=Math.max(.65,targetZoom-.1);if(event.key==='0')reset();wake();
    },inputOptions);
    renderer.domElement.addEventListener('webglcontextlost',event => {event.preventDefault();contextLost=true;if(raf)cancelAnimationFrame(raf);raf=0;sceneFailure('图形上下文已中断，可重新加载三维场景。');});
    const observer = new ResizeObserver(resize);observer.observe(host);
    const onVisibility = () => {if(active()){resize();wake();}else if(raf){cancelAnimationFrame(raf);raf=0;}};
    window.addEventListener('port-module-shown',onVisibility);document.addEventListener('visibilitychange',onVisibility);window.addEventListener('resize',resize);
    host.dataset.renderer='webgl';
    $('hd-scene-loading').hidden=true;$('hd-scene-fallback').hidden=true;resize();theme(state.theme);renderer.shadowMap.needsUpdate=true;wake();
    const api={update(frame){update(frame);wake();},focus,reset,theme,redraw:wake,get focusKey(){return focusKey;},view(key){if(key==='overview')reset();else focus(key);},layer(value){yard.visible=value!=='transport'&&value!=='services';quay.visible=value!=='transport';transport.visible=value==='overview'||value==='transport';facilities.visible=value==='overview'||value==='services';routeLines.visible=value==='transport';host.dataset.layer=value;renderer.shadowMap.needsUpdate=true;wake();},dispose(){disposed=true;inputController.abort();if(raf)cancelAnimationFrame(raf);observer.disconnect();window.removeEventListener('port-module-shown',onVisibility);document.removeEventListener('visibilitychange',onVisibility);window.removeEventListener('resize',resize);scene.traverse(object => {object.geometry?.dispose();if(object.material){(Array.isArray(object.material)?object.material:[object.material]).forEach(material=>material.dispose());}});textures.forEach(t=>t.dispose());renderer.dispose();renderer.domElement.remove();}};
    return api;
  }

  function sceneFailure(message) {
    $('hd-scene-loading').hidden=true;$('hd-scene-fallback').hidden=false;put('hd-scene-error',message);
    document.querySelectorAll('.hd-scene-label').forEach(node=>{node.style.opacity='0';});
  }
  async function initScene() {
    if(initialization)return initialization;
    initialization=(async()=>{
      try {
        $('hd-scene-loading').hidden=false;$('hd-scene-fallback').hidden=true;
        state.scene?.dispose();state.scene=null;
        const layout=await bootstrap();
        if(!layout.world || !Array.isArray(layout.yards) || !Array.isArray(layout.lanes))throw new Error('港区布局接口未返回有效几何。');
        state.layout=layout;
        try {state.scene=await createScene(layout);}
        catch(renderError) {
          const {createCompatibleScene}=await import('/static/scene-compatible.js?v=20261031-1');
          state.scene=createCompatibleScene($('hd-scene-host'),'port',key=>inspect(key,true),layout);
          $('hd-scene-loading').hidden=true;$('hd-scene-fallback').hidden=true;
          document.querySelectorAll('.hd-scene-label').forEach(node=>{node.style.opacity='0';});
        }
        state.scene.layer(state.layer);if(state.frame)state.scene.update(state.frame);
      } catch(error) {sceneFailure(`三维场景未能加载（${error.message || 'WebGL 不可用'}）。指标与二维入口仍可使用。`);}
    })().finally(()=>{initialization=null;});return initialization;
  }

  async function refresh() {
    const button=$('hd-refresh');if(button.disabled)return;button.disabled=true;put('hd-data-message','');
    try {await Promise.allSettled([loadFrame(),loadProvenance(),loadCoverage(),loadAssets(),loadServices()]);if(!state.scene)await initScene();}
    finally {button.disabled=false;}
  }

  initHeader();
  $('hd-refresh').addEventListener('click',refresh);
  window.addEventListener('public-port-desk-closed',()=>state.scene?.redraw());
  $('hd-energy-asset').addEventListener('change',()=>{put('hd-energy-value','—');energyChart?.clear();loadEnergy();});
  $('hd-play').addEventListener('click',()=>pause(!state.paused));
  $('hd-reset').addEventListener('click',()=>state.scene?.reset());
  $('hd-scene-retry').addEventListener('click',initScene);
  $('hd-inspector-close').addEventListener('click',()=>{$('hd-scene-inspector').hidden=true;});
  $('hd-camera').addEventListener('change',()=>{stopTour();state.scene?.view($('hd-camera').value);});
  $('hd-theme').addEventListener('click',()=>{state.theme=state.theme==='day'?'night':'day';const button=$('hd-theme');button.textContent=state.theme==='day'?'☀ 日景':'☾ 夜景';button.setAttribute('aria-pressed',String(state.theme==='night'));state.scene?.theme(state.theme);});
  $('hd-labels-toggle').addEventListener('click',()=>{state.labels=!state.labels;$('hd-labels-toggle').setAttribute('aria-pressed',String(state.labels));state.scene?.redraw();});
  $('hd-map-reset').addEventListener('click',()=>state.scene?.reset());
  $('hd-tour').addEventListener('click',()=>{if(state.tour){stopTour();return;}state.tour=true;$('hd-tour').setAttribute('aria-pressed','true');$('hd-tour').textContent='Ⅱ 停止巡航';state.scene?.view('quay');});
  let tourIndex=0;setInterval(()=>{if(state.tour&&active()){const views=['quay','yard','energy','gate','transport'];state.scene?.view(views[++tourIndex%views.length]);}},7000);
  $('hd-history-slider').addEventListener('input',()=>{const index=Number($('hd-history-slider').value),frame=state.history[index];if(!frame)return;pause(true);state.historyIndex=index;state.frame=frame;syncHistory();renderFrame(frame,false);home.dataset.frameTimestamp=frame.source_timestamp||String(frame.ts);streamStatus(`历史留帧 ${index+1}/${state.history.length}`,'paused');});
  $('hd-history-latest').addEventListener('click',()=>{state.historyIndex=state.history.length-1;if(state.historyIndex>=0){state.frame=state.history[state.historyIndex];renderFrame(state.frame,false);}syncHistory();pause(false);});
  document.querySelectorAll('[data-hd-service]').forEach(button=>button.addEventListener('click',()=>openService(button.dataset.hdService)));
  $('hd-service-close').addEventListener('click',()=>$('hd-service-dialog').close());
  $('hd-service-route').addEventListener('click',()=>$('hd-service-dialog').close());
  $('hd-service-dialog').addEventListener('close',()=>{serviceToken++;serviceChart?.dispose();serviceChart=null;state.scene?.redraw();});
  $('hd-service-dialog').addEventListener('click',event=>{if(event.target===$('hd-service-dialog')){const rect=event.target.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)event.target.close();}});
  document.querySelectorAll('[data-hd-focus]').forEach(button=>button.addEventListener('click',()=>inspect(button.dataset.hdFocus)));
  document.querySelectorAll('[data-hd-layer]').forEach(button=>button.addEventListener('click',()=>{stopTour();setLayer(button.dataset.hdLayer);$('hd-scene-inspector').hidden=true;}));
  $('hd-fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.hd-scene-panel').requestFullscreen();}catch(_){put('hd-data-message','当前浏览器未允许全屏，仍可拖动和缩放沙盘。');}});
  document.addEventListener('fullscreenchange',()=>{$('hd-fullscreen').setAttribute('aria-label',document.fullscreenElement?'退出全屏':'全屏三维沙盘');});
  $('hd-open-evidence').addEventListener('click',()=>{const panel=$('hd-evidence');panel.open=true;panel.scrollIntoView({behavior:motionQuery.matches?'instant':'smooth',block:'start'});});
  home.addEventListener('keydown',event=>{if(event.key==='Escape'){$('hd-scene-inspector').hidden=true;}});
  const start=()=>{refresh();if(!state.scene)initScene();};
  if(window.PortModuleLoading)window.PortModuleLoading.register('home-hero',start,{key:'visual-overview',maxAgeMs:30000});else start();
  setInterval(()=>{if(active())loadFrame();},2500);
  setInterval(()=>{if(active()){loadEnergy();loadProvenance();loadServices();}},30000);
  window.addEventListener('port-module-shown',()=>{if(active() && Date.now()-state.lastGood>15000)loadFrame();energyChart?.resize();});
})();
