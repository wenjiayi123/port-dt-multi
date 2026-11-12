/* Shared read-only data audit. Each system retains its native data and contracts. */
(function(){
  'use strict';
  const specs={
    xiaoyi:{name:'舷鸥 · GullMate',snapshot:'/api/runtime/snapshot',contract:'/api/port-simulator/contract',mode:'公开数据校准实时模拟',
      replacement:'XIAOYI_PORT_DATA_MODE=live；XIAOYI_PORT_BASE_URL / XIAOYI_PORT_API_TOKEN / 允许主机配置。站点适配器保留port-ops.v1与port-realtime.v1。',
      pending:['现场TOS / AIS / EMS与实际设备、告警','现场计量校准和独立运营验收','真实船舶身份、实际ETA和生产任务回执','模型权重与知识来源继续独立校验']},
    energy:{name:'港口能碳驾驶舱',snapshot:'/api/runtime/snapshot',contract:'/api/runtime/contract',mode:'洛杉矶公开数据标定工程模拟',
      replacement:'保留telemetry-field.v1逐字段元数据与runtime合同，由TOS / EMS / BMS / BA / 气象适配器供数。正式站点切换仍使用原有集成与验收流程。',
      pending:['全港排放源与Scope 2市场法凭证','电价结算、需量响应和碳资产登记证明','现场表计校准、设备资产与工单回读','独立M&V和生产算法准入']},
    malacca:{name:'马六甲港航韧性推演',snapshot:'/api/operations/snapshot',contract:'/api/operations/contracts/telemetry',mode:'跨港公开参考标定工程模拟',
      replacement:'现场六源通过/api/operator-integration/contract的签名、单位、时效和顺序合同接入；业务前台保留port-operations.telemetry.v1。',
      pending:['马六甲现场AIS / TOS / VTS授权来源','站点计量、设备校准与运营验收','高分实时影像和授权AIS服务凭证','数字签名、现场身份与独立控制联锁']}
  };
  const finite=v=>typeof v==='number'&&Number.isFinite(v);
  const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=(v,d=2)=>finite(v)?v.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
  const valueOK=v=>v!==null&&v!==undefined&&v!==''&&(typeof v!=='number'||finite(v));
  function normalise(kind,s,c){
    if(!s||!c||typeof s!=='object'||typeof c!=='object')throw new Error('源响应不是数据对象');
    let groups=[],fields=[],sources=[],clock,site,readouts=[],quality;
    if(kind==='xiaoyi'){
      if(!s.simulation||!s.metadata||!Array.isArray(c.domains))throw new Error('舷鸥遥测/合同不匹配');
      const records={port_call:s.port_calls,ais_vts:s.ais_tracks,berth_marine_services:s.berths,equipment:s.equipment_assets,
        yard_inventory:s.yard_blocks,gate_intermodal:s.gates,energy_carbon:[s.energy],weather_tide:[s.weather_tide],safety_maintenance:s.alerts,governance:[{...s.simulation,...s.quality,...s.metadata,...s.governance}]};
      groups=c.domains.map(d=>{const rows=(records[d.id]||[]).filter(Boolean);let complete=0,nonempty=0;const missing=[];
        (d.required_fields||[]).forEach(key=>{if(rows.length&&rows.every(r=>Object.hasOwn(r,key)))complete++;else if(rows.length)missing.push(key);
          if(rows.length&&rows.every(r=>valueOK(r[key])))nonempty++;
          fields.push({id:d.id+'.'+key,unit:key.endsWith('_kw')?'kW':key.endsWith('_kwh')?'kWh':key.includes('_percent')?'%':key.endsWith('_m_mllw')?'m MLLW':key.endsWith('_m')?'m':key.endsWith('_ms')?'m/s':key.endsWith('_knots')?'kn':key.endsWith('_minutes')?'min':key.endsWith('_hours')?'h':key.endsWith('_teu')?'TEU':key.includes('_at')||key.endsWith('_time')?'含时区时间':key.endsWith('_sha256')?'SHA-256':'见原生合同',value:rows.length?rows[0][key]:null,source:d.source_system,quality:rows.length?'模拟实体字段':'无实体记录',kind:'engineering_simulation',detail:`${rows.length}条记录；${rows.length?rows.filter(r=>valueOK(r[key])).length:0}条有值`});});
        return {id:d.id,total:d.required_fields.length,present:complete,valued:nonempty,records:rows.length,missing,replacement:d.source_system};});
      sources=(c.calibration_sources||[]).map(v=>({id:v.id||v.dataset_id||v.name||v.path||'已声明来源',detail:JSON.stringify(v)}));
      clock=s.simulation.observed_at;site=s.port?.name;quality=s.quality;
      readouts=[['总需电功率',s.energy?.grid_demand_kw,'kW'],['岸电负荷',s.energy?.shore_power_kw,'kW'],['储能荷电状态',s.energy?.bess_soc_percent,'%'],['堆场占用',s.yard?.occupancy_percent,'%'],['闸口排队',s.gate?.queue_vehicles,'辆'],['岸桥作业',s.fleet_summary?.quay_cranes?.working,'台']];
    }else{
      const telemetry=kind==='energy'?s.signals:s.operationalTelemetry;
      if(!telemetry||!s.schema_version)throw new Error('原生逐字段遥测未返回');
      const flat=kind==='energy'?Object.entries(telemetry).map(([id,f])=>({id,...f})):Object.entries(telemetry).flatMap(([domain,items])=>Object.entries(items).map(([id,f])=>({id:domain+'.'+id,...f})));
      for(const f of flat){const group=f.id.split('.')[0];let row=groups.find(g=>g.id===group);if(!row){row={id:group,total:0,present:0,valued:0,records:0,missing:[],replacement:'见适配器映射'};groups.push(row);}row.total++;row.records++;
        const keyAttrs=c.required_field_attributes||c.required_metadata||s.contract?.required_metadata||['value','unit','event_time','source_type','source_id'];
        if(keyAttrs.every(k=>Object.hasOwn(f,k)))row.present++;else row.missing.push(f.id);
        if(valueOK(f.value))row.valued++;
        fields.push({id:f.id,unit:f.unit,value:f.value,source:f.source_id,quality:f.quality_status,kind:f.source_type,detail:f.source_record_time||f.event_time});
      }
      sources=kind==='energy'?[{id:s.dataset?.dataset_id,detail:JSON.stringify(s.dataset)}]:s.calibration?.datasets?.map(v=>({id:v.id,detail:JSON.stringify(v)}))||[];
      const get=(key)=>fields.find(f=>f.id===key)?.value;
      readouts=kind==='energy'?[['电网输入',get('grid.import_power_kw'),'kW'],['可用光伏',get('solar.available_power_kw'),'kW'],['储能功率（正放负充）',get('battery.power_kw'),'kW'],['岸电负荷',get('shore_power.load_kw'),'kW'],['岸桥利用率',get('equipment.quay_crane_utilization_pct'),'%'],['设备AGV作业',get('equipment.active_agvs'),'台']]:
        [['吞吐',s.kpis?.throughput_teu,'TEU / 模拟步'],['用电',s.kpis?.energy_kwh,'kWh / 模拟步'],['电力峰值',s.kpis?.peak_grid_kw,'kW'],['待泊',s.kpis?.queue_vessels,'艘'],['本步延误',s.kpis?.delay_minutes,'min'],['场景碳排',s.kpis?.carbon_tons,'tCO₂e / 模拟步']];
      clock=s.virtual_event_time||s.event_time;site=s.site_id;quality=s.quality;
    }
    return {schema:'system-data-readiness-review.v1',system:kind,site,clock,readouts,groups,fields,sources,quality,
      fieldCount:kind==='energy'&&finite(c.field_count)?c.field_count:fields.length,valuedCount:groups.reduce((n,g)=>n+g.valued,0),
      structureCount:groups.reduce((n,g)=>n+g.present,0),
      mode:s.simulation_mode===false||s.metadata?.live_data_verified===true?'外部源快照 · 核验状态另查':specs[kind].mode,
      production_authority:false,dispatch_allowed:false,boundary:'本清单检查展示数据，不授予现场验收、审计或设备控制权。空值可能表示未知、尚未发生或不适用，不能一律补0。',
      adapter_slots:kind==='energy'?Object.entries(c.source_adapter_mapping?.replace_at_site||{}).map(([id,patterns])=>({id,current:patterns.join(' · '),production_replacement:id})):kind==='malacca'?c.adapter_slots||[]:groups.map(g=>({id:g.id,current:'工程场景字段',production_replacement:g.replacement})),
      native_snapshot:s,native_contract:c};
  }
  // Export a small pure adapter for source/null/contract regression tests.
  window.SystemDataReadiness={normalise};
  const kind=document.currentScript?.dataset.system;if(!specs[kind])return;const config=specs[kind];
  let dialog,loaded=null,revision=0,view='overview',controller;
  function table(head,rows){return `<div class="sdd-table"><table><thead><tr>${head.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;}
  const stat=(name,value,note)=>`<div class="sdd-stat"><small>${esc(name)}</small><b>${esc(value)}</b><span>${esc(note)}</span></div>`;
  function display(v){return typeof v==='number'?fmt(v):typeof v==='boolean'?(v?'是':'否'):v&&typeof v==='object'?JSON.stringify(v):v??'—';}
  function render(){if(!loaded||!dialog.open)return;const d=loaded,body=dialog.querySelector('.sdd-content');dialog.querySelectorAll('[data-sdd-view]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.sddView===view)));
    dialog.querySelector('.sdd-clock').textContent=`场景时间 ${d.clock||'未提供'} · ${d.site||'未提供站点'} · ${d.mode}`;
    if(view==='overview')body.innerHTML=`<div class="sdd-stats">${stat('本次字段',d.fieldCount,'来自原生源，不跨系统套值')}${stat('有值字段',d.valuedCount,'全记录均有值；0/false有效')}${stat('字段结构齐备',d.structureCount+'/'+d.fieldCount,'空实体集合不计作结构完整')}${stat('生产执行权限','独立验收','演示齐备不等于现场接入')}</div><div class="sdd-readouts">${d.readouts.map(([n,v,u])=>stat(n,display(v),u)).join('')}</div><h3>业务域与适配器</h3>${table(['域','字段结构','有值字段','记录数','现场替换 / 缺项'],d.groups.map(g=>`<tr><td>${esc(g.id)}</td><td>${g.present} / ${g.total}</td><td>${g.valued} / ${g.total}</td><td>${g.records}</td><td>${esc(g.replacement)}${g.missing.length?`<small>未齐备：${esc(g.missing.join(' · '))}</small>`:''}</td></tr>`))}<p class="sdd-note">${kind==='xiaoyi'?'治理域从同一快照的simulation、metadata、quality和governance归一化读取，不重算源哈希。':''}结构齐备不等于实测、准确或核证；空值和空实体集合应按业务语义解释，不能一律补0。</p>`;
    if(view==='fields')body.innerHTML=`<h3>逐项数值、单位、来源和时间</h3>${table(['字段','首条实体 / 遥测值','单位','来源类别 / 质量','来源 / 时间'],d.fields.map(f=>`<tr><td>${esc(f.id)}</td><td>${esc(display(f.value))}</td><td>${esc(f.unit)}</td><td>${esc(f.kind)}<small>${esc(f.quality)}</small></td><td>${esc(f.source)}<small>${esc(f.detail)}</small></td></tr>`))}`;
    if(view==='sources')body.innerHTML=`<h3>公开依据与工程边界</h3>${d.sources.map(s=>`<details open><summary>${esc(s.id)}</summary><pre>${esc(s.detail)}</pre></details>`).join('')}<h3>现场缺项仍单独保留</h3><div class="sdd-pending">${config.pending.map(p=>`<p>${esc(p)}</p>`).join('')}</div><h3>替换数据源</h3><p>${esc(config.replacement)}</p>${table(['适配器槽','当前字段/实现','现场替换'],d.adapter_slots.map(v=>`<tr><td>${esc(v.id)}</td><td>${esc(v.current)}</td><td>${esc(v.production_replacement)}</td></tr>`))}<p class="sdd-note">算法输出需要映射为当前合同、保留模型版本和来源时间；模型现场校准、留出验证和生产门禁继续独立执行。</p><div class="sdd-links"><a href="${config.snapshot}" target="_blank" rel="noopener">原生数据接口 ↗</a><a href="${config.contract}" target="_blank" rel="noopener">原生字段合同 ↗</a></div><details><summary>质量与守恒检查</summary><pre>${esc(JSON.stringify(d.quality,null,2))}</pre></details>`;
  }
  function shell(){if(dialog)return;dialog=document.createElement('dialog');dialog.className='sdd-dialog';dialog.setAttribute('aria-label',config.name+'数据齐备检查');
    dialog.innerHTML=`<div class="sdd-head"><div><small>DATA / SOURCES / ADAPTERS</small><h2>${config.name} · 数据齐备检查</h2><p>每个系统保留自己的公开基准与工程模型</p></div><div><button type="button" data-sdd-refresh>刷新检查</button><button type="button" data-sdd-export disabled>导出快照与合同</button><button type="button" data-sdd-close aria-label="关闭数据齐备检查">×</button></div></div><div class="sdd-tabs" role="tablist" aria-label="数据齐备视图">${[['overview','数据概览'],['fields','字段与单位'],['sources','依据与换源']].map(([key,name])=>`<button type="button" role="tab" data-sdd-view="${key}" aria-selected="${view===key}">${name}</button>`).join('')}</div><div class="sdd-content" role="tabpanel"></div><div class="sdd-clock" role="status"></div>`;
    document.body.append(dialog);dialog.addEventListener('close',()=>{revision++;controller?.abort();loaded=null;});dialog.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.hasAttribute('data-sdd-close'))dialog.close();if(b.dataset.sddView){view=b.dataset.sddView;render();}if(b.hasAttribute('data-sdd-refresh'))open();if(b.hasAttribute('data-sdd-export')&&loaded){const url=URL.createObjectURL(new Blob([JSON.stringify(loaded,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=kind+'-data-readiness.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}});
  }
  async function open(){shell();if(!dialog.open)dialog.showModal();controller?.abort();controller=new AbortController();const current=controller,token=++revision;loaded=null;dialog.querySelector('[data-sdd-export]').disabled=true;dialog.querySelector('.sdd-content').innerHTML='<div class="sdd-empty">读取本系统原生快照与字段合同…</div>';dialog.querySelector('.sdd-clock').textContent='正在核验本次来源';
    const timer=setTimeout(()=>current.abort(),12000);try{const [s,c]=await Promise.all([config.snapshot,config.contract].map(async url=>{const r=await fetch(url,{cache:'no-store',signal:current.signal});if(!r.ok)throw new Error(`${url} HTTP ${r.status}`);return r.json();}));if(token!==revision||!dialog.open)return;loaded=normalise(kind,s,c);render();dialog.querySelector('[data-sdd-export]').disabled=false;}
    catch(e){if(token!==revision||!dialog.open)return;dialog.querySelector('.sdd-content').innerHTML=`<div class="sdd-empty"><b>本次数据尚未取得</b><span>${esc(e.name==='AbortError'?'读取超时':e.message)}</span><span>旧指标与导出已清除，可刷新重试。</span></div>`;dialog.querySelector('.sdd-clock').textContent='源不可用；没有换成静态模拟数字。';}finally{clearTimeout(timer);}}
  const button=document.createElement('button');button.type='button';button.className='sdd-launcher';button.textContent='数据齐备检查';button.setAttribute('aria-label','打开数据齐备检查');button.addEventListener('click',open);document.body.append(button);
  window.addEventListener('pagehide',()=>{revision++;controller?.abort();loaded=null;});
})();
