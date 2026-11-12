(() => {
  'use strict';
  if (window.__demoEvidenceInstalled) return;
  window.__demoEvidenceInstalled = true;
  const script = document.currentScript;
  const dataUrl = script?.dataset.evidenceUrl || new URL('demo-evidence.json', script?.src || location.href).href;
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let panel, payload, selected = 0, opener, request;
  const labels = {algorithm:'具体算法',datasets:'数据集名称、数量与来源',state:'State · 状态（模型看到什么）',reward:'Reward · 奖励（怎样判断好坏）',actions:'Action · 动作（模型能决定什么）',outputs:'输出清单与验证结果',training:'训练记录与复现参数',sources:'证据文件与 SHA-256',constraints:'硬约束与适用边界'};
  const keyLabels = {id:'数据/字段 ID',label:'名称',name:'名称',title:'名称',rows:'记录数',row_count:'记录数',recordCount:'记录数',rawRecordCount:'原始记录数',source:'来源',source_url:'来源网址',sourceUrl:'来源网址',source_type:'来源类型',evidence_tier:'证据级别',evidenceLevel:'证据级别',train_rows:'训练行数',validation_rows:'验证行数',test_rows:'测试行数',blind_test_rows:'保留测试行数',independent_source_observations:'独立来源观测数',interaction_steps:'仿真交互步数',real_world_training_rows:'实船训练记录数',simulation_only:'仅仿真',measured:'现场实测',status:'状态',formula:'奖励公式',weights:'奖励权重',features:'状态字段',fields:'输出字段',path:'证据路径',sha256:'SHA-256'};
  function valueHtml(value, depth = 0) {
    if (value == null || value === '') return '<span class="de-muted">未登记（缺证据）</span>';
    if (typeof value !== 'object') {
      const str=String(value);
      if (/^https?:\/\//.test(str) && !/\s/.test(str)) return `<a href="${escape(str)}" target="_blank" rel="noopener noreferrer">${escape(str)}</a>`;
      if (str.length > 900) return `<details><summary>展开完整代码 / 原始明细（${str.length} 字符）</summary><pre>${escape(str)}</pre></details>`;
      return `<span>${escape(str)}</span>`;
    }
    if (Array.isArray(value)) {
      if (!value.length) return '<span class="de-muted">未登记（缺证据）</span>';
      if (value.every(v => typeof v !== 'object')) return `<ol class="de-list">${value.map(v => `<li>${valueHtml(v,depth+1)}</li>`).join('')}</ol>`;
      if (depth > 1 || value.length > 20) return `<details><summary>展开 ${value.length} 项完整明细</summary><pre>${escape(JSON.stringify(value,null,2))}</pre></details>`;
      return value.map(v=>`<div class="de-sub">${valueHtml(v,depth+1)}</div>`).join('');
    }
    if (depth > 2) return `<details><summary>展开 ${Object.keys(value).length} 个字段</summary><pre>${escape(JSON.stringify(value,null,2))}</pre></details>`;
    return `<dl>${Object.entries(value).map(([k,v])=>`<div><dt>${escape(keyLabels[k] || k)}</dt><dd>${valueHtml(v,depth+1)}</dd></div>`).join('')}</dl>`;
  }
  function build() {
    if (panel) return;
    const style=document.createElement('style');
    style.textContent=`.demo-evidence-panel{box-sizing:border-box;position:fixed;z-index:12040;left:320px;right:18px;top:18px;bottom:18px;display:flex;flex-direction:column;background:#081b30;color:#e5f2ff;border:1px solid #3181a7;border-radius:16px;box-shadow:0 20px 80px #000b;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Noto Sans SC",sans-serif}.demo-evidence-panel[hidden]{display:none!important}.de-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 20px;border-bottom:1px solid #28516b}.de-head h2{margin:0;font-size:20px;color:#79e6ef}.de-head p{margin:3px 0;color:#b3cadb;font-size:12px}.de-head button,.de-tabs button{color:#e5f2ff;background:#13334e;border:1px solid #39617d;border-radius:8px;padding:8px 12px;cursor:pointer}.de-tools{display:flex;gap:10px;align-items:center;padding:10px 20px}.de-tools input{flex:1;min-width:50px;padding:9px;background:#071525;color:#fff;border:1px solid #39617d;border-radius:8px}.de-layout{display:grid;grid-template-columns:minmax(180px,23%) 1fr;min-height:0;flex:1}.de-tabs{overflow:auto;padding:8px 12px;display:flex;flex-direction:column;gap:7px}.de-tabs button{text-align:left;white-space:normal;line-height:1.5}.de-tabs button[aria-selected=true]{border-color:#66ead7;background:#125064;color:#fff}.de-main{overflow:auto;padding:12px 22px 24px;min-width:0;overscroll-behavior:contain}.de-main h3{margin:0;font-size:21px;color:#fff}.de-status{color:#ffda8e;margin:7px 0 16px}.de-fields{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}.de-field{padding:13px;border:1px solid #274d68;border-radius:10px;min-width:0;background:#0b233a}.de-field h4{color:#7ae7d4;margin:0 0 9px;font-size:15px}.de-field dl{margin:0}.de-field dl>div{margin:4px 0 8px}.de-field dt{color:#91b5ce;font-size:12px}.de-field dd{margin:1px 0 0;overflow-wrap:anywhere;white-space:pre-wrap}.de-field pre{font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:400px;overflow:auto}.de-field details{margin:5px 0}.de-field summary{cursor:pointer;color:#8ce5f3}.de-field a{color:#8ce5f3}.de-list{padding-left:22px;margin:0}.de-list li{padding:2px 0;overflow-wrap:anywhere}.de-sub{border-bottom:1px solid #234861;padding-bottom:6px;margin-bottom:9px}.de-brief{font-size:13px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 10px;max-height:150px;overflow:auto}.de-muted{color:#fdbe81}.de-wide{grid-column:1/-1}.de-error{padding:24px;color:#ffc7b0}.de-error button{margin:15px}.de-empty{padding:20px;color:#fdbe81}@media(max-width:1000px){.demo-evidence-panel{left:12px;right:12px;top:12px;bottom:52vh}.de-fields{grid-template-columns:1fr}.de-layout{grid-template-columns:32% 1fr}.de-head{padding:8px 12px}.de-head h2{font-size:16px}.de-head p{display:none}.de-main{padding:8px}.de-field{padding:9px}}`;
    document.head.append(style);panel=document.createElement('section');panel.className='demo-evidence-panel';panel.hidden=true;panel.setAttribute('role','dialog');panel.setAttribute('aria-label','已训练场景算法与数据清单');
    panel.innerHTML='<header class="de-head"><div><h2>已训练场景 · 算法与数据清单</h2><p data-de-summary></p></div><button type="button" data-de-close aria-label="关闭场景清单">关闭 ×</button></header><div class="de-tools"><input data-de-search aria-label="搜索场景、算法或数据集" placeholder="搜索场景、算法或数据集"><span data-de-count></span><button type="button" data-de-download>下载清单</button></div><div class="de-layout"><nav class="de-tabs" aria-label="已训练场景列表"></nav><main class="de-main" aria-live="polite"></main></div>';
    document.body.append(panel);panel.querySelector('[data-de-close]').onclick=close;panel.querySelector('[data-de-search]').oninput=renderTabs;
    panel.querySelector('[data-de-download]').onclick=()=>{if(!payload)return;const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.href=url;a.download='trained-scene-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  }
  function renderTabs(){
    const q=panel.querySelector('[data-de-search]').value.trim().toLowerCase();const matching=payload.scenarios.map((scene,index)=>({scene,index})).filter(({scene})=>!q||JSON.stringify(scene).toLowerCase().includes(q));
    if (!matching.some(v=>v.index===selected) && matching.length) selected=matching[0].index;
    panel.querySelector('[data-de-count]').textContent=`${matching.length} / ${payload.scenarios.length} 个场景`;
    const nav=panel.querySelector('.de-tabs');nav.innerHTML=matching.map(({scene,index})=>`<button type="button" data-de-scene="${index}" aria-selected="${index===selected}">${escape(scene.name)}</button>`).join('');
    nav.querySelectorAll('[data-de-scene]').forEach(button=>button.onclick=()=>{selected=Number(button.dataset.deScene);renderTabs();});
    if(matching.length)renderScene();else panel.querySelector('.de-main').innerHTML='<p class="de-empty">没有匹配的已训练场景。请缩短搜索词。</p>';
  }

  function brief(key,value){
    if(key==='algorithm'){
      if(typeof value==='string') return value;
      if(Array.isArray(value)) return value.join(' · ');
      return [value.trained_rl?.join(' · '),value.method,value.base_model?.model_id,value.selected,typeof value.learning_method==='string'?value.learning_method:'',value.candidates?.map(v=>v.id).join(' · ')].filter(Boolean).join(' · ') || JSON.stringify(value);
    }
    if(key==='datasets'){
      const rows=[];
      const scan=(v,depth=0)=>{
        if(!v||typeof v!=='object'||depth>3)return;
        if(Array.isArray(v)){v.forEach(x=>scan(x,depth+1));return;}
        const name=v.label||v.title||v.name||v.id||v.dataset_id;
        const counts=Object.entries(v).filter(([k,x])=>typeof x==='number'&&/rows|count|steps|examples|recordCount|observations|episodes/i.test(k)).map(([k,x])=>`${keyLabels[k]||k}: ${x.toLocaleString()}`);
        if(name||counts.length) rows.push([name,...counts].filter(Boolean).join(' · '));
        if(v.description)rows.push(v.description);
        if(v.evidence_tier||v.evidenceLevel||v.source_type)rows.push(v.evidence_tier||v.evidenceLevel||v.source_type);
        if(!name||!counts.length)Object.values(v).forEach(x=>scan(x,depth+1));
      };
      scan(value);return [...new Set(rows)].slice(0,5).join('\n') || '完整数据血缘与计数见下方明细';
    }
    if(key==='state'||key==='actions'){
      const fields=Array.isArray(value)?value:value.fields||value.policy_action||value.feature_names||value.actions;
      if(Array.isArray(fields))return `${value.dimensions||fields.length} 项：`+fields.map(v=>typeof v==='string'?v:v.label||v.description||v.id||JSON.stringify(v)).join(' · ');
      return typeof value==='string'?(value.length>250?'代码契约与完整顺序已登记，请展开查看':value):JSON.stringify(value);
    }
    if(key==='reward')return typeof value==='string'?value: value.formula||value.training_objective||value.base||value.clarification||JSON.stringify(value.weights||value.components||value.terms||value);
    if(key==='outputs')return (value.business_metric_fields||value.fields||value.trajectory_fields||Object.keys(value.test_summary||value)).join(' · ');
    return '';
  }

  function renderScene(){
    const s=payload.scenarios[selected];panel.dataset.sceneId=s.id;
    panel.querySelector('.de-main').innerHTML=`<h3>${escape(s.name)}</h3><p class="de-status">${escape(typeof s.status==='string'?s.status:JSON.stringify(s.status))}</p><div class="de-fields">${Object.entries(labels).filter(([key])=>key in s).map(([key,label])=>`<section class="de-field ${['sources','training','constraints'].includes(key)?'de-wide':''}" data-evidence-field="${key}"><h4>${label}</h4>${['sources','training','constraints'].includes(key)?'':`<p class="de-brief">${escape(brief(key,s[key]))}</p>`}<details><summary>展开${label}完整清单</summary>${valueHtml(s[key])}</details></section>`).join('')}</div>`;
    panel.querySelector('.de-main').scrollTop=0;
  }
  async function open(trigger){
    build();opener=trigger || document.activeElement;panel.hidden=false;panel.dataset.state='loading';
    panel.querySelector('.de-main').innerHTML='<p>正在读取本地训练证据…</p>';
    try {
      if(!payload){request ||= fetch(dataUrl,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();}).finally(()=>request=null);payload=await request;}
      if(payload.schema!=='demo-trained-scenarios.v1'||!payload.scenarios?.length)throw Error('清单为空或格式无效');
      panel.querySelector('[data-de-summary]').textContent=`${payload.system} · 证据快照 ${new Date(payload.generated_at).toLocaleString()} · ${payload.boundary}`;
      renderTabs();panel.dataset.state='ready';panel.dataset.demoState='complete';document.documentElement.dataset.demoEvidenceReady='true';
      if(trigger){trigger.dataset.demoState='complete';trigger.setAttribute('aria-expanded','true');}
      window.portDtDemoReceipt?.('trained-evidence',{target:'.demo-evidence-panel',message:`已显示 ${payload.scenarios.length} 个已训练场景，逐项可核验算法、数据、状态、奖励、动作和输出。`});
      window.dispatchEvent(new CustomEvent('demo-evidence-ready',{detail:{count:payload.scenarios.length}}));
    } catch(error){panel.dataset.state='error';panel.querySelector('.de-main').innerHTML=`<div class="de-error">证据读取失败：${escape(error.message)}。本步未标为成功。<button type="button" data-de-retry>重试读取</button></div>`;panel.querySelector('[data-de-retry]').onclick=()=>open(trigger);}
  }
  function close(){if(panel)panel.hidden=true;document.documentElement.dataset.demoEvidenceReady='false';opener?.focus?.({preventScroll:true});}
  document.addEventListener('click',event=>{const trigger=event.target.closest('[data-demo-evidence-open]');if(trigger)void open(trigger);});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&panel&&!panel.hidden)close();});
  window.addEventListener('demo-evidence-open',()=>void open());window.addEventListener('demo-evidence-close',close);
})();
