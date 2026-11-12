/* Read-only demo data. Lazy per module; no timer, observer, model or gateway writes. */
(function(){
  'use strict';
  if(window.PortDemoData)return;
  const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite=n=>typeof n==='number'&&Number.isFinite(n);
  const fmt=(n,d=1)=>finite(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
  const labels={public_fact:'公开事实',resource_simulation:'资源推演',engineering_parameter:'工程设定',schedule_derived:'计划派生',engineering_derived:'工程派生',historical_reanalysis:'历史再分析',adapter_snapshot:'外部快照',snapshot_derived:'快照计算'};
  const routes={'exec-cockpit':'operations','deep-section':'operations','kpi-section':'energy','realtime-section':'energy','esg-section':'reports','cmp-section':'reports','ext-section':'operations'};
  let cached=null,cacheTime=0,pending=null,sitePending=null,active=null,revision=0;
  async function request(url){const c=new AbortController(),timer=setTimeout(()=>c.abort(),15000);try{const r=await fetch(url,{cache:'no-store',signal:c.signal});if(!r.ok)throw new Error(`HTTP ${r.status}`);return await r.json();}finally{clearTimeout(timer);}}
  function load(refresh=false){
    if(pending)return pending;
    if(!refresh&&cached&&Date.now()-cacheTime<60000)return Promise.resolve(cached);
    if(refresh)cached=null;
    pending=request('/api/public-port-demo/workspace').then(d=>{if(d.schema!=='port-dt-demo-workspace.v1'||!Array.isArray(d.metrics)||!d.snapshot?.profile?.timezone||!d.window||!d.annual||!d.coverage||!Array.isArray(d.site_pending))throw new Error('展示数据契约不匹配');cached=d;cacheTime=Date.now();return d;}).catch(e=>{cached=null;throw e;}).finally(()=>pending=null);
    return pending;
  }
  function card(name,value,unit){return `<div class="dd-stat"><small>${esc(name)}</small><strong>${esc(value)}</strong><span>${esc(unit)}</span></div>`;}
  const when=(t,tz)=>new Date(t).toLocaleString('zh-CN',{timeZone:tz,hour12:false,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
  function metric(d,id){return d.metrics.find(m=>m.id===id)||{value:null,name:id,unit:'未提供'};}
  function metricCards(d,ids){return `<div class="dd-stats">${ids.map(id=>{const m=metric(d,id);return card(m.name,fmt(m.value),`${m.unit} · ${labels[m.evidence_class]||m.evidence_class||''}`);}).join('')}</div>`;}
  function table(head,rows){return `<div class="dd-table"><table><thead><tr>${head.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;}
  function renderView(host,d,view){
    const s=d.snapshot,o=s.operating,tz=s.profile.timezone,w=d.window;
    host.classList.add('dd-view');
    if(view==='operations'){
      const calls=s.calls.filter(c=>Date.parse(c.planned_completion)>Date.parse(d.operations.plan_window_start)&&Date.parse(c.planned_berth_time)<Date.parse(d.operations.plan_window_end));
      host.innerHTML=metricCards(d,['berths_busy','qc_busy','qc_moves_per_hour','mean_service_hours','mean_wait_minutes','yard_occupancy_pct'])+
        `<div class="dd-columns"><div class="dd-box"><h4>泊位与配桥 · 当前场景</h4>${Array.from({length:o.berths_total},(_,i)=>{const c=s.calls.find(c=>c.berth_index===i&&c.phase==='working');return `<div class="dd-berth"><b>B${String(i+1).padStart(2,'0')}</b><span>${c?esc(c.call_id):'本时刻未分配作业'}</span><i><em style="width:${c?Math.max(0,Math.min(100,c.progress*100)):0}%"></em></i><small>${c?fmt(c.progress*100)+'%':'—'}</small></div>`;}).join('')}<p>工程设定与计划派生，不代表实际船时效率。AGV / 轨道吊活动按岸桥工作量映射。</p></div><div class="dd-box"><h4>窗口内计划装卸 · ${calls.length} 航次</h4>${table(['航次 / 泊位','开始 → 完成 '+tz,'箱量 TEU'],calls.slice(0,9).map(c=>`<tr><td>${esc(c.call_id)}<small>${esc(c.berth_id)}</small></td><td>${esc(when(c.planned_berth_time,tz))} → ${esc(when(c.planned_completion,tz))}</td><td>${fmt(c.cargo_teu,0)}</td></tr>`))}<p>窗口为当前前后12小时；箱量/服务时间是作业计划。标定小时吞吐是独立流量输入，两者不充当已结算实绩。</p></div></div>`;
    }else if(view==='energy'){
      const groups=s.asset_groups||[],max=Math.max(1,...groups.map(g=>g.kW));
      host.innerHTML=metricCards(d,['energy_kwh','electricity_carbon_kg','peak_kw','energy_kwh_per_teu','carbon_kg_per_teu','throughput_teu'])+
        `<div class="dd-columns"><div class="dd-box"><h4>当前机群功率构成 · kW</h4>${groups.length?groups.map(g=>`<div class="dd-load"><span>${esc(g.label)}</span><i><em style="width:${Math.max(0,Math.min(100,g.kW/max*100))}%"></em></i><b>${fmt(g.kW)}</b></div>`).join(''):'<p>适配器未提供分项。</p>'}<p>${d.groups_complete?'分项与总功率守恒。':'分项未覆盖总功率，不显示占比。'} 内置分项是工程分摊，不是现场子表或安装证明。</p></div><div class="dd-box"><h4>分时汇总 · ${esc(tz)}</h4>${w.available?table(['时段','小时数','用电 MWh','电力碳排 t','箱量 TEU'],w.shifts.map(r=>`<tr><td>${esc(r.label)}</td><td>${r.hours}</td><td>${fmt(r.energy_kwh/1000,2)}</td><td>${fmt(finite(r.electricity_carbon_kg)?r.electricity_carbon_kg/1000:null,2)}</td><td>${fmt(r.throughput_teu,0)}</td></tr>`)):`<p>${esc(w.reason)}</p>`}<p>${esc(w.boundary||'缺失输入不补零。')} 电价与碳因子沿用工程输入，现场以合同/核验因子替换。</p><div class="dd-boundary">Scope 1：未核算<br>燃油、燃气及范围3未包含；不能将电力碳排标作全港总碳排。</div></div></div>`;
    }else if(view==='reports'){
      const a=d.annual;
      host.innerHTML=`<div class="dd-report-heading"><div><h4>${a.available?'2025 月度能碳演示报表':'当前展示源窗口报表'}</h4><p>按同一小时序列累计 · 不混用全上海港与单码头规模</p></div><span class="dd-label">工程测算 · 未核证</span></div>`+
        (a.available?`<div class="dd-stats dd-four">${card('全年标定吞吐',fmt(a.totals.throughput_teu,0),'TEU · 采用公开800万下界')}${card('全年工程电量',fmt(a.totals.energy_kwh/1e6,2),'GWh · 按吞吐规模缩放')}${card('全年电力碳排',fmt(a.totals.electricity_carbon_kg/1000,1),'tCO₂e · 不含Scope 1')}${card('完整小时序列',fmt(a.totals.hours,0),'条 · 2025 UTC年')}</div>`+table(['月份 UTC','小时数','标定吞吐 TEU','用电 MWh','电力碳排 tCO₂e','kWh/TEU','kgCO₂e/TEU'],a.months.map(r=>`<tr><td>${r.month}</td><td>${r.hours}</td><td>${fmt(r.throughput_teu,0)}</td><td>${fmt(r.energy_kwh/1000,2)}</td><td>${fmt(r.electricity_carbon_kg/1000,2)}</td><td>${fmt(r.energy_kwh_per_teu,3)}</td><td>${fmt(r.carbon_kg_per_teu,3)}</td></tr>`))+`<p class="dd-note">${esc(a.boundary)} 日内分时使用场景时区，年度按UTC年，共8,760小时；两种边界均明确标注。</p>`:
        `<div class="dd-boundary">${esc(a.reason)}</div>`+metricCards(d,['energy_kwh','electricity_carbon_kg','throughput_teu','energy_kwh_per_teu']));
    }else if(view==='coverage'){
      host.innerHTML=`<div class="dd-stats dd-four">${card('本次展示字段',d.coverage.available+' / '+d.coverage.total,'非现场接入率')}${card('已声明公开来源',fmt(s.profile.sources.length,0),'条 · 日期见来源服务台')}${card('仍需现场补齐',d.site_pending.length,'类 · 下表列明原因')}${card('生产执行权限','未授予','独立验收与审批')}</div><div class="dd-box"><h4>展示字段、来源类型与替换位置</h4>${table(['数据项','当前值','来源类型','后续替换','口径'],d.metrics.map(m=>`<tr><td>${esc(m.name)}</td><td>${fmt(m.value)} ${esc(m.unit)}</td><td>${esc(labels[m.evidence_class]||m.evidence_class)}</td><td>${esc(m.replacement)}</td><td class="dd-wrap">${esc(m.detail)}</td></tr>`))}</div>${pendingMarkup(d)}<div class="dd-boundary">${d.not_applicable.map(x=>`${esc(x.name)}：${esc(x.reason)}`).join('<br>')}</div>`;
    }
  }
  function pendingMarkup(d){return `<div class="dd-box"><h4>保留的现场缺项</h4><div class="dd-pending-grid">${d.site_pending.map(p=>`<div><b>${esc(p.name)}</b><p>${esc(p.reason)}</p></div>`).join('')}</div></div>`;}
  async function site(host,d,ctx){
    host.innerHTML='<div class="dd-empty">核验原有现场适配器…</div>';
    try{if(!sitePending)sitePending=request('/api/v3/site-integration/readiness').finally(()=>sitePending=null);const r=await sitePending;if(active!==ctx||ctx.view!=='site')return;
      host.innerHTML=`<div class="dd-stats dd-four">${card('现场适配器',fmt(r.configured_adapter_count,0)+' / '+fmt(r.adapter_count,0),'实际后端配置')}${card('实时来源核验',fmt(r.live_verified_adapter_count,0),'已核验适配器')}${card('演示填充','独立展示层','不写入现场准入')}${card('现场数据与模型','分步替换','原有校准/影子/验收保留')}</div><div class="dd-box">${table(['适配器','配置','实时核验','缺项'],(r.adapters||[]).map(a=>`<tr><td>${esc(a.adapter_id)}</td><td>${a.configured?'已配置':'未配置'}</td><td>${a.live_data_verified?'通过':'待核验'}</td><td class="dd-wrap">${esc((a.blockers||[]).join(' · '))}</td></tr>`))}</div>${pendingMarkup(d)}`;
    }catch(e){if(active===ctx&&ctx.view==='site')host.innerHTML=`<div class="dd-empty">现场状态读取失败：${esc(e.message)}。点击本页刷新重试；未沿用旧状态。</div>`;}
  }
  function draw(ctx){
    if(active!==ctx||!ctx.data)return;
    ctx.host.querySelectorAll('[data-dd-view]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.ddView===ctx.view)));
    if(ctx.view==='site')site(ctx.content,ctx.data,ctx);else renderView(ctx.content,ctx.data,ctx.view);
  }
  function mount(){
    const id=document.body.dataset.moduleView;const token=++revision;
    active?.host.remove();active=null;
    if(!routes[id])return;
    const root=document.getElementById(id),board=root?.querySelector('.mv-workspace');if(!board)return;
    const host=document.createElement('section');host.className='dd-workspace';host.setAttribute('data-i18n-skip','');host.setAttribute('aria-label','演示业务数据与现场接入');
    host.innerHTML=`<div class="dd-head"><div><span class="dd-kicker">BUSINESS DATA / REPLACEABLE PRESENTATION FEED</span><h3>演示业务数据</h3><p>公开资源规模 + 可复现工程场景；下方原有现场状态与模型证据保留各自口径。</p></div><div class="dd-actions"><button type="button" data-dd-refresh>刷新数据</button><button type="button" data-dd-export disabled>导出数据清单</button></div></div><div class="dd-tabs" role="tablist" aria-label="演示数据视图">${[['operations','作业与资源'],['energy','能碳分项'],['reports','月度报表'],['coverage','数据齐备清单'],['site','现场接入状态']].map(([key,name])=>`<button type="button" role="tab" data-dd-view="${key}" aria-selected="${key===routes[id]}">${name}</button>`).join('')}</div><div class="dd-content" role="tabpanel"></div><div class="dd-foot" role="status">正在读取统一展示数据源…</div>`;
    board.after(host);const ctx={host,content:host.querySelector('.dd-content'),view:routes[id],data:null,token,request:0};active=ctx;
    async function refresh(force){
      const seq=++ctx.request;ctx.data=null;ctx.content.innerHTML='<div class="dd-empty">读取演示业务数据…</div>';host.querySelector('[data-dd-export]').disabled=true;host.querySelector('[data-dd-refresh]').disabled=true;
      try{const d=await load(force);if(active!==ctx||seq!==ctx.request||token!==revision)return;ctx.data=d;draw(ctx);host.querySelector('[data-dd-export]').disabled=false;host.querySelector('.dd-foot').textContent=`${d.source_mode} · ${when(d.snapshot.source_timestamp,d.snapshot.profile.timezone)} ${d.snapshot.profile.timezone} · 展示字段 ${d.coverage.available}/${d.coverage.total} · 非实时现场数据`;}
      catch(e){if(active!==ctx||seq!==ctx.request)return;ctx.content.innerHTML=`<div class="dd-empty"><b>本次展示源不可用</b><p>${esc(e.name==='AbortError'?'请求超时':e.message)} · 点击刷新数据重试</p></div>`;host.querySelector('.dd-foot').textContent='本次数据未返回，旧指标与导出已清除；原有模型和现场网关继续独立工作。';}
      finally{if(active===ctx&&seq===ctx.request)host.querySelector('[data-dd-refresh]').disabled=false;}
    }
    host.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.ddView){ctx.view=b.dataset.ddView;draw(ctx);}if(b.hasAttribute('data-dd-refresh'))refresh(true);if(b.hasAttribute('data-dd-export')&&ctx.data){const blob=new Blob([JSON.stringify(ctx.data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='port-demo-data-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}});
    refresh(false);
  }
  window.PortDemoData={load,renderView};
  window.addEventListener('port-visual-workspace-ready',mount);
  window.addEventListener('port-module-shown',()=>{if(!routes[document.body.dataset.moduleView]){revision++;active?.host.remove();active=null;}});
  window.addEventListener('pagehide',()=>{revision++;active?.host.remove();active=null;cached=null;});
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();
