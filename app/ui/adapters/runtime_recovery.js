(function(){
  if(window.__portDtRuntimeRecoveryInstalled) return;
  window.__portDtRuntimeRecoveryInstalled = true;

  const HEALTH_URL = "/health/live";
  const MODULES = {
    b: {label:"堆场照明", section:"yl-section", status:"b-v3-status", method:"b-training-method", values:["b-rew-base","b-rew-pol","b-kpi-gain","b-kpi-dkwh","b-kpi-peak","b-kpi-ulx"]},
    c: {label:"HVAC 冷站", section:"hvac-section", status:"c-v3-status", method:"c-training-method", values:["c-kpi-chws","c-kpi-sat","c-kpi-sp","c-kpi-peak","c-kpi-kwh","c-kpi-mse"]},
    d: {label:"岸电储能", section:"sbess-section", status:"d-v3-status", method:"d-training-method", values:["d-kpi-soc","d-kpi-save","d-kpi-peak","d-kpi-chg","d-kpi-dis","d-kpi-carbon"]},
    be:{label:"场内储能", section:"be-section", status:"be-v3-status",method:"be-training-method",values:["be-kpi-soc","be-kpi-dkwh","be-kpi-save","be-kpi-peak","be-kpi-chg","be-kpi-dis"]},
    f: {label:"场桥 / 轨道吊", section:"yc-section", status:"f-v3-status", method:"f-training-method", values:["f-kpi-work","f-kpi-peak","f-kpi-util","f-kpi-lat","f-kpi-save","f-kpi-co2"]}
  };
  const INITIAL_STATE = /(加载中|读取中|核验中|正在核验|正在读取|等待.*回执)/;
  let consecutiveFailures = 0;
  let confirmedOffline = false;
  let evidenceFailureObserved = false;
  let reloadScheduled = false;
  let probeInFlight = false;
  let intervalID = null;
  let provenanceSnapshot = null;
  let globalHideTimer = null;
  const moduleLoads = new Map();
  const moduleRequests = new Map();
  const deferredRenders = new Map();
  let renderObserver = null;
  const installedAt = Date.now();
  const STARTUP_GRACE_MS = 20000;
  const HEALTH_TIMEOUT_MS = 8000;

  function byId(id){ return document.getElementById(id); }

  function ensureBanner(){
    let banner = byId("port-dt-runtime-recovery");
    if(banner) return banner;
    banner = document.createElement("aside");
    banner.id = "port-dt-runtime-recovery";
    banner.hidden = true;
    banner.setAttribute("role", "status");
    banner.setAttribute("aria-live", "polite");
    banner.innerHTML = `
      <span class="port-dt-runtime-recovery-dot" aria-hidden="true"></span>
      <div class="port-dt-runtime-recovery-copy">
        <strong>本地数据加载中</strong>
        <small>当前页面证据尚未请求</small>
        <div class="port-dt-global-load-track" role="progressbar" aria-label="本地证据加载进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-valuetext="尚未请求"><span></span></div>
      </div>
      <button type="button" hidden>立即重试</button>`;
    const style = document.createElement("style");
    style.id = "port-dt-runtime-recovery-style";
    style.textContent = `
      #port-dt-runtime-recovery{position:fixed;left:50%;top:14px;transform:translateX(-50%);z-index:3550;display:flex;align-items:center;gap:10px;width:min(680px,calc(100vw - 32px));padding:11px 13px;border:1px solid rgba(248,113,113,.58);border-radius:13px;background:linear-gradient(145deg,rgba(76,18,31,.97),rgba(33,18,38,.98));box-shadow:0 18px 44px rgba(0,0,0,.42);color:#fff;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
      #port-dt-runtime-recovery[hidden]{display:none}
      #port-dt-runtime-recovery.loading{border-color:rgba(56,189,248,.54);background:linear-gradient(145deg,rgba(7,48,87,.97),rgba(8,30,57,.98))}
      #port-dt-runtime-recovery.recovered{border-color:rgba(52,211,153,.62);background:linear-gradient(145deg,rgba(5,69,53,.97),rgba(7,37,48,.98))}
      #port-dt-runtime-recovery .port-dt-runtime-recovery-dot{width:9px;height:9px;flex:0 0 auto;border-radius:50%;background:#fb7185;box-shadow:0 0 12px rgba(251,113,133,.82)}
      #port-dt-runtime-recovery.loading .port-dt-runtime-recovery-dot{background:#38bdf8;box-shadow:0 0 12px rgba(56,189,248,.82)}
      #port-dt-runtime-recovery.recovered .port-dt-runtime-recovery-dot{background:#34d399;box-shadow:0 0 12px rgba(52,211,153,.82)}
      #port-dt-runtime-recovery .port-dt-runtime-recovery-copy{min-width:0;flex:1}#port-dt-runtime-recovery strong{display:block;font-size:13px}#port-dt-runtime-recovery small{display:block;margin-top:2px;color:#fecdd3;font-size:11px;line-height:1.45}#port-dt-runtime-recovery.loading small{color:#bae6fd}#port-dt-runtime-recovery.recovered small{color:#bbf7d0}
      #port-dt-runtime-recovery button{min-height:32px;border:1px solid rgba(255,255,255,.22);border-radius:9px;background:rgba(15,23,42,.72);color:#fff;padding:0 11px;font-weight:800;cursor:pointer}
      .port-dt-global-load-track{height:5px;margin-top:7px;overflow:hidden;border-radius:999px;background:rgba(30,64,175,.42)}.port-dt-global-load-track span{display:block;width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,#38bdf8,#60a5fa,#34d399);transition:width .22s ease-out}.port-dt-global-load-track span.is-indeterminate{width:34%;animation:portDtLoadSweep 1.05s ease-in-out infinite}
      .port-dt-module-load-progress{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:5px 10px;align-items:center;margin:8px 0 10px;padding:8px 10px;border:1px solid rgba(59,130,246,.42);border-radius:9px;background:rgba(5,17,38,.72);color:#bfdbfe;font-size:11px}
      .port-dt-module-load-progress[hidden]{display:none}.port-dt-module-load-progress strong{font-size:11px;color:#dbeafe}.port-dt-module-load-progress small{color:#93c5fd;font-variant-numeric:tabular-nums}
      .port-dt-module-load-track{grid-column:1/-1;height:5px;overflow:hidden;border-radius:999px;background:rgba(30,64,175,.35)}.port-dt-module-load-track span{display:block;width:34%;height:100%;border-radius:inherit;background:linear-gradient(90deg,#38bdf8,#60a5fa,#34d399);animation:portDtLoadSweep 1.05s ease-in-out infinite}
      .port-dt-module-load-progress[data-state="complete"]{border-color:rgba(52,211,153,.38);color:#bbf7d0}.port-dt-module-load-progress[data-state="complete"] strong{color:#d1fae5}.port-dt-module-load-progress[data-state="complete"] small{color:#86efac}.port-dt-module-load-progress[data-state="complete"] .port-dt-module-load-track span{width:100%;animation:none;background:#34d399}
      .port-dt-module-load-progress[data-state="failed"]{border-color:rgba(248,113,113,.52);color:#fecaca}.port-dt-module-load-progress[data-state="failed"] strong,.port-dt-module-load-progress[data-state="failed"] small{color:#fecaca}.port-dt-module-load-progress[data-state="failed"] .port-dt-module-load-track span{width:100%;animation:none;background:#fb7185}
      @keyframes portDtLoadSweep{0%{transform:translateX(-105%)}50%{transform:translateX(95%)}100%{transform:translateX(295%)}}
      @media(prefers-reduced-motion:reduce){.port-dt-module-load-track span{animation-duration:2.4s}}
      @media(max-width:720px){#port-dt-runtime-recovery{top:8px;align-items:flex-start;flex-wrap:wrap}#port-dt-runtime-recovery button{margin-left:19px}}
    `;
    document.head.appendChild(style);
    document.body.appendChild(banner);
    banner.querySelector("button")?.addEventListener("click", ()=>probeHealth(true));
    return banner;
  }

  function ensureModuleProgress(key){
    const module = MODULES[key];
    if(!module) return null;
    let root = byId(`port-dt-progress-${key}`);
    if(root) return root;
    const status = byId(module.status);
    if(!status) return null;
    root = document.createElement("div");
    root.id = `port-dt-progress-${key}`;
    root.className = "port-dt-module-load-progress";
    root.hidden = true;
    root.dataset.state = "idle";
    root.innerHTML = `
      <strong>${module.label}证据</strong>
      <small>等待读取</small>
      <div class="port-dt-module-load-track" role="progressbar" aria-label="${module.label}证据加载进度" aria-valuemin="0" aria-valuemax="100" aria-valuetext="等待读取"><span></span></div>`;
    status.insertAdjacentElement("afterend", root);
    return root;
  }

  function clearModuleTimer(key){
    const state = moduleLoads.get(key);
    if(state?.timer) window.clearInterval(state.timer);
  }

  function evidenceLoadSummary(){
    const current = window.PortModuleLoading?.current() || window.PortModuleNavigation?.current || document.body?.dataset.moduleView || "home-hero";
    const keys = Object.keys(MODULES).filter(key=>MODULES[key].section===current);
    const complete = keys.filter(key=>moduleLoads.get(key)?.status === "complete");
    const loading = keys.filter(key=>moduleLoads.get(key)?.status === "loading");
    const failed = keys.filter(key=>moduleLoads.get(key)?.status === "failed");
    const idle = keys.filter(key=>!moduleLoads.has(key));
    return {total:keys.length, complete, loading, failed, idle};
  }

  function evidenceLoadsPending(){
    return Array.from(moduleLoads.values()).some(state=>state.status === "loading");
  }

  function setGlobalTrack(banner, percent, text, indeterminate=false){
    const track = banner.querySelector(".port-dt-global-load-track");
    const fill = track?.querySelector("span");
    if(!track || !fill) return;
    fill.classList.toggle("is-indeterminate", indeterminate);
    if(indeterminate){
      track.removeAttribute("aria-valuenow");
      fill.style.removeProperty("width");
    }else{
      const normalized = Math.max(0, Math.min(100, Number(percent) || 0));
      track.setAttribute("aria-valuenow", String(normalized));
      fill.style.width = `${normalized}%`;
    }
    track.setAttribute("aria-valuetext", text);
  }

  function updateGlobalProgress(){
    const summary = evidenceLoadSummary();
    const banner = ensureBanner();
    if(globalHideTimer){
      window.clearTimeout(globalHideTimer);
      globalHideTimer = null;
    }
    // Unopened energy pages have no requests and are not homepage work. Do not
    // count them as complete, or make the homepage wait for a fictitious 0/5.
    if(!summary.total){
      if(!confirmedOffline) banner.hidden=true;
      return;
    }
    if(summary.failed.length || confirmedOffline) return;
    const percent = Math.round((summary.complete.length / summary.total) * 100);
    banner.hidden = false;
    banner.querySelector("button").hidden = true;
    if(summary.complete.length === summary.total){
      banner.classList.remove("loading");
      banner.classList.add("recovered");
      banner.querySelector("strong").textContent = "当前页面证据已就绪";
      banner.querySelector("small").textContent = `${summary.complete.length}/${summary.total} 个当前页面证据模块读取完成`;
      setGlobalTrack(banner, 100, `已完成 ${summary.complete.length}/${summary.total}`);
      globalHideTimer = window.setTimeout(()=>{
        if(!confirmedOffline) banner.hidden = true;
      }, 1200);
      window.setTimeout(()=>probeHealth(false), 120);
      return;
    }
    banner.classList.remove("recovered");
    banner.classList.add("loading");
    banner.querySelector("strong").textContent = "当前页面证据加载中";
    const active = summary.loading.map(key=>MODULES[key].label);
    const detail = active.length ? `·正在读取 ${active.join("、")}` : "·尚未请求，等待当前页面加载任务启动";
    banner.querySelector("small").textContent = `已完成 ${summary.complete.length}/${summary.total} ${detail}`;
    setGlobalTrack(banner, percent, `已完成 ${summary.complete.length}/${summary.total}`);
  }

  function updateElapsed(key){
    const state = moduleLoads.get(key);
    const root = ensureModuleProgress(key);
    if(!state || !root || root.dataset.state !== "loading") return;
    const elapsed = Math.max(0, (performance.now() - state.startedAt) / 1000);
    const detail = root.querySelector("small");
    if(detail) detail.textContent = `读取后端证据 · ${elapsed.toFixed(1)} 秒`;
    const track = root.querySelector('[role="progressbar"]');
    track?.setAttribute("aria-valuetext", `正在读取，已用 ${elapsed.toFixed(1)} 秒`);
  }

  function markModuleLoading(key){
    const module = MODULES[key];
    const root = ensureModuleProgress(key);
    if(!module || !root) return;
    clearModuleTimer(key);
    const state = {startedAt:performance.now(), timer:null, status:"loading"};
    moduleLoads.set(key, state);
    root.hidden = false;
    root.dataset.state = "loading";
    root.querySelector("strong").textContent = `${module.label}证据读取中`;
    root.querySelector('[role="progressbar"]')?.removeAttribute("aria-valuenow");
    byId(module.section)?.setAttribute("aria-busy", "true");
    updateElapsed(key);
    state.timer = window.setInterval(()=>updateElapsed(key), 200);
    updateGlobalProgress();
  }

  function markModuleComplete(key){
    const module = MODULES[key];
    const state = moduleLoads.get(key);
    const root = ensureModuleProgress(key);
    if(!module || !root) return;
    clearModuleTimer(key);
    const elapsed = state ? Math.max(0, (performance.now() - state.startedAt) / 1000) : 0;
    root.hidden = false;
    root.dataset.state = "complete";
    root.querySelector("strong").textContent = `${module.label}证据已就绪`;
    root.querySelector("small").textContent = `读取完成 · ${elapsed.toFixed(1)} 秒`;
    const track = root.querySelector('[role="progressbar"]');
    track?.setAttribute("aria-valuenow", "100");
    track?.setAttribute("aria-valuetext", `读取完成，用时 ${elapsed.toFixed(1)} 秒`);
    byId(module.section)?.setAttribute("aria-busy", "false");
    const completedState = state || {startedAt:performance.now(), timer:null};
    completedState.status = "complete";
    moduleLoads.set(key, completedState);
    updateGlobalProgress();
    restoreActiveModuleAnchor();
  }

  function markModuleProgressFailed(key, error){
    const module = MODULES[key];
    const root = ensureModuleProgress(key);
    if(!module || !root) return;
    clearModuleTimer(key);
    root.hidden = false;
    root.dataset.state = "failed";
    root.querySelector("strong").textContent = `${module.label}证据读取失败`;
    root.querySelector("small").textContent = String(error?.message || error || "服务不可用").slice(0, 96);
    const track = root.querySelector('[role="progressbar"]');
    track?.removeAttribute("aria-valuenow");
    track?.setAttribute("aria-valuetext", "读取失败，可重试");
    byId(module.section)?.setAttribute("aria-busy", "false");
    const failedState = moduleLoads.get(key) || {startedAt:performance.now(), timer:null};
    failedState.status = "failed";
    moduleLoads.set(key, failedState);
  }

  function runDeferredRender(sectionId){
    if(!deferredRenders.has(sectionId)) return;
    window.requestAnimationFrame(()=>window.setTimeout(()=>{
      const section = byId(sectionId);
      // Navigation can hide a module after its intersection callback was queued.
      // Keep the draw pending until the section has a real layout again.
      if(section && !section.getClientRects().length){
        renderObserver?.observe(section);
        return;
      }
      const task = deferredRenders.get(sectionId);
      if(!task) return;
      deferredRenders.delete(sectionId);
      try{ task(); }catch(error){ console.warn(`[deferred-render:${sectionId}]`, error); }
      restoreActiveModuleAnchor();
    }, 0));
  }

  function restoreActiveModuleAnchor(){
    // The module router owns page positioning. Background evidence must not
    // navigate back to a previous module or reset the user's reading position.
    if(window.PortModuleNavigation) return;
    const activeHash = window.location.hash;
    const targetId = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    if(!targetId || !Object.values(MODULES).some(module=>module.section === targetId)) return;
    window.requestAnimationFrame(()=>window.setTimeout(()=>{
      if(window.PortModuleNavigation || window.location.hash !== activeHash) return;
      const target = byId(targetId);
      if(!target || !target.getClientRects().length) return;
      const header = document.querySelector("header");
      const offset = (header?.offsetHeight || 0) + 18;
      const top = Math.max(0, window.scrollY + target.getBoundingClientRect().top - offset);
      window.scrollTo({top, behavior:"auto"});
    }, 60));
  }

  window.__portDtDeferHeavyRender = function(sectionId, task){
    if(typeof task !== "function") return;
    const section = byId(sectionId);
    if(!section || !("IntersectionObserver" in window)){
      window.setTimeout(task, 0);
      return;
    }
    deferredRenders.set(sectionId, task);
    if(!renderObserver){
      renderObserver = new IntersectionObserver(entries=>{
        entries.forEach(entry=>{
          if(!entry.isIntersecting) return;
          renderObserver.unobserve(entry.target);
          runDeferredRender(entry.target.id);
        });
      }, {rootMargin:"320px 0px"});
    }
    const rect = section.getBoundingClientRect();
    if(section.getClientRects().length && rect.bottom >= -160 && rect.top <= window.innerHeight + 160){
      runDeferredRender(sectionId);
      return;
    }
    renderObserver.observe(section);
  };

  window.__portDtEvidenceLoadStarted = markModuleLoading;
  window.__portDtEvidenceLoadCompleted = markModuleComplete;
  window.__portDtTrackEvidenceLoad = function(key, task){
    if(moduleRequests.has(key)) return moduleRequests.get(key);
    markModuleLoading(key);
    const pending=Promise.resolve().then(task).then(result=>{
      markModuleComplete(key);
      return result;
    },error=>{
      window.__portDtEvidenceLoadFailed(key, error);
      throw error;
    }).finally(()=>{if(moduleRequests.get(key)===pending)moduleRequests.delete(key);});
    moduleRequests.set(key,pending);
    return pending;
  };

  function markModuleOffline(key){
    const module = MODULES[key];
    if(!module) return;
    const status = byId(module.status);
    const method = byId(module.method);
    if(status) status.textContent = `后端证据服务已中断 · ${HEALTH_URL} 正在自动重试`;
    if(method && INITIAL_STATE.test(method.textContent || "")) method.textContent = "证据暂不可读：等待本地后端恢复，不填充静态替代指标。";
    module.values.forEach(id=>{
      const node = byId(id);
      if(node && INITIAL_STATE.test(node.textContent || "")) node.textContent = "服务离线";
    });
  }

  function confirmOffline(markEvidenceModules){
    confirmedOffline = true;
    const banner = ensureBanner();
    banner.hidden = false;
    banner.classList.remove("recovered");
    banner.classList.toggle("loading", !markEvidenceModules);
    banner.querySelector("button").hidden = false;
    banner.querySelector("strong").textContent = markEvidenceModules ? "本地数据服务连接中断" : "本地数据服务响应较慢";
    banner.querySelector("small").textContent = markEvidenceModules
      ? "正在自动恢复；页面不会用静态数字冒充后端证据。"
      : "首屏证据核验仍在继续；正在重试健康检查。";
    setGlobalTrack(banner, 0, markEvidenceModules ? "数据服务中断" : "正在重试健康检查", !markEvidenceModules);
    if(markEvidenceModules){
      Object.keys(MODULES).forEach(markModuleOffline);
      const provenance = byId("system-provenance-badge");
      if(provenance){
        if(!provenanceSnapshot){
          provenanceSnapshot = {text:provenance.textContent, color:provenance.style.color};
        }
        provenance.textContent = "数据来源状态：本地服务离线（自动重试中）";
        provenance.style.color = "#ef4444";
      }
    }
  }

  function restoreProvenance(){
    if(!provenanceSnapshot) return;
    const provenance = byId("system-provenance-badge");
    if(provenance){
      provenance.textContent = provenanceSnapshot.text;
      provenance.style.color = provenanceSnapshot.color;
    }
    provenanceSnapshot = null;
  }

  function scheduleReload(){
    if(reloadScheduled) return;
    reloadScheduled = true;
    const banner = ensureBanner();
    banner.hidden = false;
    banner.classList.add("recovered");
    banner.classList.remove("loading");
    banner.querySelector("button").hidden = true;
    banner.querySelector("strong").textContent = "本地数据服务已恢复";
    banner.querySelector("small").textContent = "正在重新读取模型、数据哈希、盲测与准入证据…";
    window.setTimeout(()=>window.location.reload(), 650);
  }

  async function probeHealth(immediate){
    if(probeInFlight || reloadScheduled) return;
    // Only started evidence requests are business traffic. Unopened pages must
    // neither start work nor block recovery probes indefinitely.
    if(!immediate && evidenceLoadsPending()) return;
    probeInFlight = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(()=>controller.abort(), HEALTH_TIMEOUT_MS);
    try{
      const response = await fetch(HEALTH_URL, {cache:"no-store", signal:controller.signal});
      if(!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json().catch(()=>({}));
      if(payload.status !== "alive" && payload.ok !== true) throw new Error("identity mismatch");
      consecutiveFailures = 0;
      if(confirmedOffline && evidenceFailureObserved) scheduleReload();
      else if(confirmedOffline){
        confirmedOffline = false;
        const banner = ensureBanner();
        banner.hidden = true;
        restoreProvenance();
      }
    }catch(_){
      consecutiveFailures += 1;
      // A cold homepage validates many sizeable evidence bundles at once.  A
      // slow health response during that window is not itself proof that the
      // backend died.  Actual evidence fetch failures are handled immediately
      // below; polling alone needs repeated failures after the startup grace.
      if(Date.now()-installedAt >= STARTUP_GRACE_MS && consecutiveFailures >= 3){
        confirmOffline(false);
      }
    }finally{
      window.clearTimeout(timeout);
      probeInFlight = false;
    }
  }

  window.__portDtEvidenceLoadFailed = function(moduleKey, error){
    evidenceFailureObserved = true;
    markModuleProgressFailed(moduleKey, error);
    markModuleOffline(moduleKey);
    confirmOffline(true);
    probeHealth(true);
  };
  window.__portDtRuntimeRecovery = {probe:()=>probeHealth(true)};
  window.addEventListener("port-module-shown", updateGlobalProgress);

  function start(){
    ensureBanner();
    Object.keys(MODULES).forEach(ensureModuleProgress);
    updateGlobalProgress();
    probeHealth(false);
    if(!intervalID) intervalID = window.setInterval(()=>probeHealth(false), 4000);
    window.setTimeout(()=>probeHealth(true), STARTUP_GRACE_MS);
  }
  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, {once:true});
  else start();
  window.addEventListener("pageshow", ()=>probeHealth(true));
  document.addEventListener("visibilitychange", ()=>{if(!document.hidden) probeHealth(true);});
})();
