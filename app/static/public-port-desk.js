/* On-demand operational presentation. Accepted model pages are excluded. */
(function(){
  'use strict';
  if(window.__publicPortDesk)return;window.__publicPortDesk=true;
  const preserved=new Set(['train-section','yl-section','hvac-section','sbess-section','be-section','yc-section','rlops-section','strategy-exec-module']);
  const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=(n,d=0)=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
  const phaseNames={working:'装卸中',waiting:'待泊',planned:'计划',completed:'已完成'};
  let dialog,data,demoWorkspace,pending,charts=[],tabName='berths',requestId=0;
  const time=t=>new Date(t).toLocaleString('zh-CN',{timeZone:data?.profile?.timezone||'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
  function allowed(){return location.pathname!=='/rl-panel'&&!preserved.has(document.body.dataset.moduleView);}
  function disposeCharts(){charts.forEach(c=>c.dispose());charts=[];}
  function makeChart(id,option){if(!window.echarts)return;const plot=echarts.init(document.getElementById(id));plot.setOption({animation:false,color:['#78d7c2','#8cafdf','#d9b681'],textStyle:{fontFamily:'system-ui',color:'#90afbf'},tooltip:{trigger:'axis',confine:true,backgroundColor:'#142b38',borderColor:'#3b5967',textStyle:{color:'#cce2ec',fontSize:11}},...option});charts.push(plot);}
  function stat(name,value,unit){return `<div class="ppd-stat"><span>${esc(name)}</span><b>${esc(value)}</b><small>${esc(unit)}</small></div>`;}
  async function load(force=false){
    if(window.PortDemoData){const w=await window.PortDemoData.load(force);demoWorkspace=w;return w.snapshot;}
    if(pending)return pending;
    pending=(async()=>{const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);try{const response=await fetch('/api/public-port-demo/snapshot',{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error(`HTTP ${response.status}`);const next=await response.json();if(next.schema!=='port-dt-presentation-snapshot.v1')throw new Error('数据格式不匹配');return next;}finally{clearTimeout(timer);}})().finally(()=>{pending=null;});return pending;
  }
  function shell(){
    if(dialog)return;
    dialog=document.createElement('dialog');dialog.className='ppd-dialog';dialog.setAttribute('aria-labelledby','ppd-title');dialog.setAttribute('data-i18n-skip','');
    dialog.innerHTML='<div class="ppd-head"><div><div class="ppd-kicker">PUBLIC PARAMETERS / OPERATIONS SCENARIO</div><h2 id="ppd-title">港口作业与服务台</h2><p>洋山四期 · 公开参数标定演示 · 非现场遥测</p></div><div class="ppd-head-actions"><button type="button" data-ppd-action="refresh">刷新快照</button><button type="button" data-ppd-action="export">导出数据</button><button type="button" data-ppd-action="close" aria-label="关闭港口作业服务台">×</button></div></div><div class="ppd-tabs" role="tablist" aria-label="港口服务台视图"><button role="tab" type="button" data-ppd-action="tab" data-tab="berths" aria-selected="true">泊位与航次</button><button role="tab" type="button" data-ppd-action="tab" data-tab="equipment" aria-selected="false">设备与作业</button><button role="tab" type="button" data-ppd-action="tab" data-tab="energy" aria-selected="false">能碳与环境</button><button role="tab" type="button" data-ppd-action="tab" data-tab="reports" aria-selected="false">月度报表</button><button role="tab" type="button" data-ppd-action="tab" data-tab="coverage" aria-selected="false">数据齐备清单</button><button role="tab" type="button" data-ppd-action="tab" data-tab="sources" aria-selected="false">公开依据与替换</button></div><div class="ppd-content" id="ppd-content" role="tabpanel"></div><div class="ppd-footer"><span id="ppd-clock">等待读取场景时间</span><span>演示推演 / 生产控制未授权</span></div>';
    document.body.append(dialog);
    dialog.addEventListener('click',event=>{const button=event.target.closest('[data-ppd-action]');if(!button)return;const action=button.dataset.ppdAction;
      if(action==='close')dialog.close();
      if(action==='tab'){tabName=button.dataset.tab;render();}
      if(action==='refresh')open(true);
      if(action==='export'&&data){const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='yangshan-public-presentation-snapshot.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    });
    dialog.addEventListener('close',()=>{requestId++;disposeCharts();window.dispatchEvent(new CustomEvent('public-port-desk-closed'));});
    window.addEventListener('resize',()=>{if(dialog.open)charts.forEach(c=>c.resize());});
  }
  async function open(force=false){
    if(!allowed())return;shell();if(!dialog.open)dialog.showModal();
    const token=++requestId;data=null;demoWorkspace=null;disposeCharts();dialog.querySelector('[data-ppd-action="export"]').disabled=true;document.getElementById('ppd-clock').textContent='正在核验快照时间';dialog.querySelector('.ppd-content').innerHTML='<div class="ppd-empty">正在读取公开标定演示快照…</div>';
    try{const next=await load(force);if(token!==requestId||!dialog.open)return;data=next;dialog.querySelector('[data-ppd-action="export"]').disabled=false;render();}
    catch(error){if(token!==requestId||!dialog.open)return;dialog.querySelector('.ppd-content').innerHTML=`<div class="ppd-empty"><b>演示数据尚未读取</b><p>${esc(error.message)} · 请使用“刷新快照”重试。</p></div>`;}
  }
  function render(){
    if(!data||!dialog.open)return;disposeCharts();const body=dialog.querySelector('.ppd-content'),o=data.operating,p=data.profile,now=Date.parse(data.source_timestamp);
    dialog.querySelectorAll('[data-ppd-action="tab"]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===tabName)));
    dialog.querySelector('.ppd-head p').textContent=`${p.terminal||p.name||'港口'} · ${data.mode==='public_parameter_calibrated_demo'?'公开参数标定演示':'外部展示快照'} · 来源核验另行验收`;
    document.getElementById('ppd-clock').textContent=`${new Date(data.source_timestamp).getUTCFullYear()}年 ${time(data.source_timestamp)} ${p.timezone} · 展示场景时间`;
    if(['reports','coverage'].includes(tabName)){
      if(demoWorkspace&&window.PortDemoData)window.PortDemoData.renderView(body,demoWorkspace,tabName);else body.innerHTML='<div class="ppd-empty">报表组件未加载，请刷新页面。</div>';
    }else if(tabName==='berths'){
      const start=now-12*3600000,end=now+12*3600000;
      const visible=data.calls.filter(c=>Date.parse(c.planned_completion)>start&&Date.parse(c.eta)<end);
      const rows=Array.from({length:o.berths_total},(_,i)=>`<div class="ppd-gantt-row"><b>B${String(i+1).padStart(2,'0')}</b><div class="ppd-gantt-track">${visible.filter(c=>c.berth_index===i).map(c=>{const left=Math.max(0,(Date.parse(c.planned_berth_time)-start)/(end-start)*100),right=Math.min(100,(Date.parse(c.planned_completion)-start)/(end-start)*100);return `<span class="ppd-call ${esc(c.phase)}" style="left:${left}%;width:${Math.max(0,right-left)}%" title="${esc(c.call_id)} · ${time(c.planned_berth_time)} → ${time(c.planned_completion)} · ${fmt(c.cargo_teu)} TEU">${esc(c.call_id.replace('DEMO-YS4-','演示 '))}</span>`;}).join('')}<i class="ppd-now"></i></div></div>`).join('');
      body.innerHTML=`<div class="ppd-stats">${stat('作业泊位',`${o.berths_busy} / ${o.berths_total}`,'资源约束推演')}${stat('岸桥分配',`${o.qc_busy} / ${o.qc_total}`,'台 · 推演配桥')}${stat('24h 标定箱量',fmt(o.daily_throughput_teu),'TEU · 工程场景')}${stat('当前作业航次',fmt(o.working_call_count),'推演航次 · 无真实船名')}</div><div class="ppd-block"><div class="ppd-block-head"><h3>泊位作业时间窗</h3><span>${time(new Date(start).toISOString())} → ${time(new Date(end).toISOString())}</span></div><div class="ppd-gantt">${rows}</div><div class="ppd-gantt-legend"><span>已完成</span><span>装卸中</span><span>计划</span><span>中线为场景当前时刻</span></div></div><div class="ppd-table-wrap"><table><thead><tr><th>推演航次</th><th>泊位</th><th>状态</th><th>计划靠泊 ${esc(p.timezone)}</th><th>计划完成 ${esc(p.timezone)}</th><th>装卸量 TEU</th><th>进度</th></tr></thead><tbody>${visible.map(c=>`<tr><td>${esc(c.call_id)}</td><td>${esc(c.berth_id)}</td><td>${phaseNames[c.phase]}</td><td>${time(c.planned_berth_time)}</td><td>${time(c.planned_completion)}</td><td>${fmt(c.cargo_teu)}</td><td><span class="ppd-progress"><i style="width:${c.progress*100}%"></i></span>${fmt(Math.min(c.phase==='working'?99.9:100,c.progress*100),1)}%</td></tr>`).join('')}</tbody></table></div><p class="ppd-note">航次 ID、靠泊时窗与进度由资源约束模型生成，船期不是实际 ETA。配桥不超过 29 台；同一泊位前后航次不重叠。</p>`;
    }else if(tabName==='equipment'){
      const grid=(name,busy,total,code)=>`<div class="ppd-block"><div class="ppd-block-head"><h3>${name}</h3><span>${busy} 作业 / ${total-busy} 其余 · 共 ${total}</span></div><div class="ppd-equipment-grid ${code==='AGV'?'dense':''}">${Array.from({length:total},(_,i)=>`<span class="${i<busy?'busy':''}" title="演示 ${code}-${i+1} · ${i<busy?'工作量映射为作业':'其余'}">${code==='AGV'?'':String(i+1).padStart(2,'0')}</span>`).join('')}</div></div>`;
      body.innerHTML=`<div class="ppd-stats">${stat('公开岸桥规模',fmt(o.qc_total),'2026-08 公开资料')}${stat('公开轨道吊规模',fmt(o.rmg_total),'2026-08 公开资料')}${stat('公开 AGV 规模',fmt(o.agv_total),'运营方与2026资料互证')}${stat('演示泊位占用',fmt(o.berth_occupancy_pct,1)+'%','从推演航次计算')}</div><div class="ppd-equipment-columns">${grid('岸桥 QC',o.qc_busy,o.qc_total,'QC')}${grid('轨道吊 RMG',o.rmg_busy,o.rmg_total,'RMG')}${grid('自动导引车 AGV',o.agv_busy,o.agv_total,'AGV')}</div><p class="ppd-note">总规模有公开依据；“作业/其余”是由岸桥工作量分配到设备群的演示映射，不表示现场故障或真实 ECS 调度。设备编号为演示编号。</p>`;
    }else if(tabName==='energy'){
      const e=data.environment;body.innerHTML=`<div class="ppd-stats">${stat('24h 标定用电',fmt(o.daily_energy_kwh/1000,1),'MWh · 工程派生')}${stat('24h 工程碳排',fmt(o.daily_carbon_kg/1000,2),'tCO₂e · 电力范围')}${stat('单位吞吐能耗',fmt(o.energy_kwh_per_teu,3),'kWh/TEU · 同窗分母')}${stat('单位吞吐碳排',fmt(o.carbon_kg_per_teu,3),'kgCO₂e/TEU · 非审计台账')}</div><div class="ppd-chart-columns"><div class="ppd-block"><h3>标定负荷与吞吐</h3><div class="ppd-chart" id="ppd-load-chart"></div></div><div class="ppd-block"><h3>公开历史海况</h3><div class="ppd-chart" id="ppd-weather-chart"></div></div></div><div class="ppd-stats ppd-weather">${stat('风速',fmt(e.wind_speed_mps,2),'m/s · 历史再分析')}${stat('浪高',fmt(e.wave_height_m,2),'m · 历史再分析')}${stat('潮位',fmt(e.tide_m,2),'m · 原序列参考面')}${stat('气温',fmt(e.ambient_c,1),'°C · 历史再分析')}</div><p class="ppd-note">能耗按公开吞吐规模缩放仓库现有工程序列，排放因子和电价为现有工程输入；未充当洋山实测电表或结算数据。气象海况保留历史再分析来源时间。</p>`;
      const series=data.series,xAxis={type:'category',data:series.map(r=>time(r.timestamp).slice(-5)),axisLabel:{fontSize:10,color:'#7fa0af'},axisTick:{show:false}},yAxis=(name,position='left')=>({type:'value',name,position,scale:true,axisLabel:{fontSize:10,color:'#7fa0af'},splitLine:{lineStyle:{color:'#263e49'}}});
      makeChart('ppd-load-chart',{legend:{top:3,textStyle:{color:'#8cabbc',fontSize:10}},grid:{left:55,right:55,top:47,bottom:32},xAxis,yAxis:[yAxis('kW'),yAxis('TEU/h','right')],series:[{name:'工程负荷',type:'line',symbol:'none',data:series.map(r=>r.power_kw)},{name:'标定吞吐',type:'line',symbol:'none',yAxisIndex:1,data:series.map(r=>r.throughput_teu_per_hour)}]});
      makeChart('ppd-weather-chart',{legend:{top:3,textStyle:{color:'#8cabbc',fontSize:10}},grid:{left:48,right:45,top:47,bottom:32},xAxis,yAxis:[yAxis('m/s'),yAxis('m','right')],series:[{name:'风速',type:'line',symbol:'none',data:series.map(r=>r.wind_speed_mps)},{name:'潮位',type:'line',symbol:'none',yAxisIndex:1,data:series.map(r=>r.tide_m)}]});
    }else{
      const names={berths:'泊位',quay_length_m:'岸线长度',land_area_m2:'陆域面积',average_berth_depth_m:'公开平均泊位水深',design_annual_teu:'设计通过能力',annual_throughput_lower_bound_teu:'2025 公开年吞吐下界',qc_count:'岸桥规模',rmg_count:'轨道吊规模',agv_count:'AGV 规模',qc_record_moves_per_hour_lower_bound:'最快单机作业纪录下界',daily_record_teu_lower_bound:'最高日吞吐纪录下界'};
      body.innerHTML=`<div class="ppd-source-layout"><div class="ppd-block"><h3>公开事实与日期</h3><div class="ppd-table-wrap"><table><thead><tr><th>参数</th><th>公开数值</th><th>日期 / 来源</th></tr></thead><tbody>${Object.entries(p.facts).map(([k,f])=>{const source=p.sources.find(s=>s.id===f.source_id);return `<tr><td>${esc(names[k]||k)}</td><td>${esc(f.comparison||'')}${fmt(f.value,2)} ${esc(f.unit)}</td><td><a href="${esc(source?.url)}" target="_blank" rel="noopener">${esc(f.as_of||source?.published_at)} · ${esc(source?.publisher)} ↗</a></td></tr>`;}).join('')}</tbody></table></div></div><div class="ppd-block"><h3>明确的工程参数</h3>${Object.entries(p.scenario_parameters).filter(([k])=>k!=='replay_start').map(([k,v])=>`<div class="ppd-assumption"><b>${esc(k)} = ${esc(v.value)}</b><p>${esc(v.basis)}</p></div>`).join('')}<details><summary>计算公式与哈希</summary><pre>${esc(JSON.stringify({formulas:data.formulas,provenance:data.provenance},null,2))}</pre></details></div></div><div class="ppd-block ppd-replace"><h3>落地时替换展示数据源</h3><p>对接程序输出相同字段的 JSON 快照，然后配置 <code>PORT_DT_PRESENTATION_SNAPSHOT=/绝对路径/snapshot.json</code>。TOS 提供航次和作业，ECS 提供设备状态，EMS 提供电表；界面继续复用。</p><a href="/api/public-port-demo/replacement-contract" target="_blank" rel="noopener">查看字段映射契约 ↗</a><p class="ppd-note">这会替换展示数据；模型仍需按现场工况校准、重训或验证，生产权限继续通过原有独立网关验收。源配置损坏时会报错，不静默回填演示数据。</p></div>`;
    }
  }
  function install(){
    if(dialog?.open)dialog.close();if(!allowed())return;
    const target=document.body.dataset.moduleView==='home-hero'?document.querySelector('.hd-heading-tools .hd-source-pill'):document.querySelector('.mv-workspace .mv-tools .mv-pill');
    if(!target||target.dataset.ppdEntry)return;
    const button=document.createElement('button');button.type='button';button.className=target.className+' ppd-entry';button.dataset.ppdEntry='true';button.textContent='港口作业服务台 ↗';button.setAttribute('aria-label','打开港口作业服务台');button.addEventListener('click',()=>open(false));target.replaceWith(button);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',install,{once:true});else install();
  window.addEventListener('port-module-shown',install);
  window.addEventListener('pagehide',disposeCharts);
})();
