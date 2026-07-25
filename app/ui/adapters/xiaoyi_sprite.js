(function(){
  if (window.__xiaoyiSpriteInstalled) return;
  window.__xiaoyiSpriteInstalled = true;

  const SPRITE_VERSION = "2026-07-12-maritime-speech-v2";
  const STORAGE_KEY = "xiaoyi_sprite_position_v1";
  const defaultMessage = "我在这里。输入“开始训练碳排最低目标”这类指令，我会先说明参数，再带你进入 RL 面板确认。";

  function isRlTrainingCommand(text){
    const q = String(text || "").replace(/\s+/g, "");
    return /(小懿)?(请|帮我|现在)?(开始|启动|执行|训练).*(RL|rl|强化学习|策略|模型|目标|能耗|碳排|电费|峰值|岸电|AGV|agv|泊位|安全|韧性|低风险)/.test(q)
      || /(RL|rl|强化学习).*(开始|启动|训练)/.test(q);
  }

  function actionText(data){
    const rec = data?.recommendation || {};
    const cfg = rec.config || data?.will_execute?.backend_request?.body?.config || {};
    const objective = cfg.objective_label || rec.objective_label || "综合最优";
    const algo = String(cfg.algorithm || "SAC").toUpperCase();
    const reason = rec.reason || "已根据训练目标选择一组适合操作员确认的保守参数。";
    return `已识别：${objective}。推荐算法：${algo}。${reason}`;
  }

  function guidedHomeUrl(data){
    const finalUrl = new URL(data?.will_execute?.open_url || "/rl-panel?action=start_rl_training&from=xiaoyi&confirm=prompt", window.location.origin);
    const cfg = data?.recommendation?.config || data?.will_execute?.backend_request?.body?.config || {};
    const guide = new URL("/", window.location.origin);
    guide.searchParams.set("xiaoyi_route", "rl_training");
    guide.searchParams.set("next", finalUrl.pathname + finalUrl.search);
    guide.searchParams.set("objective", cfg.objective || "multi_objective");
    guide.searchParams.set("objective_label", cfg.objective_label || "综合最优");
    guide.searchParams.set("algorithm", cfg.algorithm || "sac");
    guide.searchParams.set("recommendation_title", data?.recommendation?.title || "推荐训练参数");
    return guide.toString();
  }

  function safeJsonParse(value){
    try { return JSON.parse(value || ""); } catch (_) { return null; }
  }

  function clamp(value, min, max){
    return Math.max(min, Math.min(max, value));
  }

  function createStyle(){
    const style = document.createElement("style");
    style.id = "xiaoyi-sprite-style";
    style.textContent = `
      .xiaoyi-sprite-root{position:fixed;right:24px;bottom:24px;z-index:3600;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,"Noto Sans",sans-serif;color:#eaf2ff;touch-action:none}
      .xiaoyi-sprite-orb{width:96px;height:136px;border:0;background:transparent;padding:0;cursor:grab;position:relative;overflow:visible}
      .xiaoyi-sprite-orb:active{cursor:grabbing}
      .xiaoyi-sprite-character{position:absolute;inset:0;width:100%;height:100%;display:block;object-fit:contain;pointer-events:none;user-select:none;transform-origin:50% 82%;filter:drop-shadow(0 13px 17px rgba(0,0,0,.42)) drop-shadow(0 0 8px rgba(34,211,238,.22));animation:xiaoyiCharacterIdle 3.4s ease-in-out infinite}
      .xiaoyi-sprite-orb:hover .xiaoyi-sprite-character{filter:drop-shadow(0 15px 20px rgba(0,0,0,.46)) drop-shadow(0 0 11px rgba(34,211,238,.38));animation-duration:2.1s}
      .xiaoyi-sprite-speech{position:absolute;right:72px;top:-28px;width:224px;min-height:94px;display:grid;align-content:center;gap:5px;padding:14px 17px 13px;border:1px solid rgba(31,181,255,.78);border-radius:17px 17px 5px 17px;background:linear-gradient(145deg,rgba(7,48,87,.97),rgba(4,28,58,.98));box-shadow:0 16px 38px rgba(0,0,0,.34),inset 0 1px 0 rgba(143,225,255,.12),0 0 18px rgba(14,165,233,.12);text-align:left;pointer-events:none;transition:opacity .2s ease,visibility .2s ease;animation:xiaoyiSpeechGlow 3.4s ease-in-out infinite}
      .xiaoyi-sprite-speech::after{content:"";position:absolute;right:20px;bottom:-13px;width:22px;height:15px;background:linear-gradient(135deg,rgba(5,39,75,.98) 0 52%,transparent 53%);clip-path:polygon(0 0,100% 0,0 100%);filter:drop-shadow(-1px 1px 0 rgba(31,181,255,.72))}
      .xiaoyi-sprite-speech strong{font-size:16px;line-height:1.2;font-weight:900;letter-spacing:.2px;color:#f4fbff;text-shadow:0 0 12px rgba(125,211,252,.18);white-space:nowrap}
      .xiaoyi-sprite-speech small{font-size:12px;line-height:1.25;font-weight:750;color:#a9d8f5;white-space:nowrap}
      .xiaoyi-sprite-wave{height:19px;display:flex;align-items:center;gap:3px;margin-top:2px}
      .xiaoyi-sprite-wave i{width:3px;height:7px;border-radius:999px;background:#32c5ff;box-shadow:0 0 7px rgba(50,197,255,.82);animation:xiaoyiSpeechWave 1.15s ease-in-out infinite}
      .xiaoyi-sprite-wave i:nth-child(2),.xiaoyi-sprite-wave i:nth-child(8){height:13px;animation-delay:.1s}.xiaoyi-sprite-wave i:nth-child(3),.xiaoyi-sprite-wave i:nth-child(7){height:18px;animation-delay:.2s}.xiaoyi-sprite-wave i:nth-child(4),.xiaoyi-sprite-wave i:nth-child(6){height:11px;animation-delay:.3s}.xiaoyi-sprite-wave i:nth-child(5){height:16px;animation-delay:.4s}.xiaoyi-sprite-wave i:nth-child(9){height:9px;animation-delay:.5s}
      .xiaoyi-sprite-root.open .xiaoyi-sprite-speech{opacity:0;visibility:hidden;animation:none}
      .xiaoyi-sprite-root[data-page="/rl-panel"] .xiaoyi-sprite-speech{opacity:0;visibility:hidden;animation:none}
      .xiaoyi-sprite-root[data-page="/rl-panel"] .xiaoyi-sprite-orb:is(:hover,:focus-visible) .xiaoyi-sprite-speech{opacity:1;visibility:visible}
      .xiaoyi-sprite-panel{position:absolute;right:0;bottom:146px;width:min(360px,calc(100vw - 28px));border:1px solid rgba(125,211,252,.30);border-radius:18px;background:linear-gradient(180deg,rgba(7,16,31,.98),rgba(10,20,42,.98));box-shadow:0 24px 70px rgba(0,0,0,.46),inset 0 1px 0 rgba(255,255,255,.05);padding:14px;display:none;touch-action:auto}
      .xiaoyi-sprite-root.open .xiaoyi-sprite-panel{display:block}
      .xiaoyi-sprite-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
      .xiaoyi-sprite-title{font-size:14px;font-weight:900;color:#f8fbff;line-height:1.35}
      .xiaoyi-sprite-sub{font-size:12px;color:#94b8dd;line-height:1.55;margin-top:3px}
      .xiaoyi-sprite-close{width:28px;height:28px;border-radius:9px;border:1px solid rgba(148,163,184,.22);background:#0b1426;color:#cfe0ff;cursor:pointer;font-weight:900}
      .xiaoyi-sprite-log{margin-top:10px;padding:10px;border-radius:12px;border:1px solid rgba(148,163,184,.16);background:rgba(9,20,38,.76);font-size:12px;line-height:1.6;color:#cfe0ff;min-height:54px;white-space:pre-line}
      .xiaoyi-sprite-form{display:grid;gap:8px;margin-top:10px}
      .xiaoyi-sprite-input{width:100%;min-height:72px;resize:vertical;border:1px solid rgba(96,165,250,.30);border-radius:12px;background:#07111f;color:#eaf2ff;padding:10px;outline:none;font:inherit;font-size:13px;line-height:1.45}
      .xiaoyi-sprite-input:focus{border-color:#60a5fa;box-shadow:0 0 0 2px rgba(96,165,250,.16)}
      .xiaoyi-sprite-actions{display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap}
      .xiaoyi-sprite-btn{min-height:34px;border-radius:10px;border:1px solid rgba(96,165,250,.28);background:linear-gradient(180deg,#2563eb,#1d4ed8);color:#fff;padding:0 11px;font-weight:850;cursor:pointer;font-size:12px}
      .xiaoyi-sprite-btn.secondary{background:#13233d;color:#cfe0ff}
      .xiaoyi-sprite-btn:disabled{opacity:.55;cursor:not-allowed}
      .xiaoyi-sprite-hint{font-size:11px;color:#8aa6d8;line-height:1.45}
      @keyframes xiaoyiCharacterIdle{0%,100%{transform:translateY(0) rotate(-.4deg) scale(1)}50%{transform:translateY(-6px) rotate(.6deg) scale(1.015)}}
      @keyframes xiaoyiSpeechGlow{0%,100%{box-shadow:0 16px 38px rgba(0,0,0,.34),inset 0 1px 0 rgba(143,225,255,.12),0 0 14px rgba(14,165,233,.10)}50%{box-shadow:0 18px 42px rgba(0,0,0,.38),inset 0 1px 0 rgba(143,225,255,.16),0 0 23px rgba(14,165,233,.22)}}
      @keyframes xiaoyiSpeechWave{0%,100%{transform:scaleY(.55);opacity:.55}50%{transform:scaleY(1);opacity:1}}
      @media(max-width:720px){.xiaoyi-sprite-root{right:16px;bottom:16px}.xiaoyi-sprite-orb{width:82px;height:118px}.xiaoyi-sprite-speech{right:60px;top:-25px;width:190px;min-height:82px;padding:12px 14px}.xiaoyi-sprite-speech strong{font-size:14px}.xiaoyi-sprite-speech small{font-size:11px}.xiaoyi-sprite-panel{bottom:128px}}
      @media(prefers-reduced-motion:reduce){.xiaoyi-sprite-character,.xiaoyi-sprite-speech,.xiaoyi-sprite-wave i{animation:none}}
    `;
    document.head.appendChild(style);
  }

  function buildWidget(){
    const root = document.createElement("div");
    root.className = "xiaoyi-sprite-root";
    root.dataset.spriteVersion = SPRITE_VERSION;
    root.dataset.page = window.location.pathname;
    root.innerHTML = `
      <div class="xiaoyi-sprite-panel" role="dialog" aria-label="小懿AI随身助手">
        <div class="xiaoyi-sprite-head">
          <div>
            <div class="xiaoyi-sprite-title">小懿AI · 随身助手</div>
            <div class="xiaoyi-sprite-sub">可拖动，随时输入命令。</div>
          </div>
          <button class="xiaoyi-sprite-close" type="button" aria-label="收起小懿AI">×</button>
        </div>
        <div class="xiaoyi-sprite-log">${defaultMessage}</div>
        <form class="xiaoyi-sprite-form">
          <textarea class="xiaoyi-sprite-input" placeholder="例如：小懿，开始训练碳排最低目标"></textarea>
          <div class="xiaoyi-sprite-actions">
            <button class="xiaoyi-sprite-btn" type="submit">执行命令</button>
            <button class="xiaoyi-sprite-btn secondary" type="button" data-open-full>打开完整问答</button>
          </div>
          <div class="xiaoyi-sprite-hint">RL 训练会先进入人工确认，不会直接生产执行。</div>
        </form>
      </div>
      <button class="xiaoyi-sprite-orb" type="button" aria-label="打开小懿海事助手">
        <span class="xiaoyi-sprite-speech" aria-hidden="true">
          <strong>您好！我是小懿AI</strong>
          <small>您的港航智能助手</small>
          <span class="xiaoyi-sprite-wave"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span>
        </span>
        <img class="xiaoyi-sprite-character" src="/static/xiaoyi_maritime_officer.png?v=20260725-q" alt="小懿Q版海事运营助手" draggable="false" />
      </button>
    `;
    document.body.appendChild(root);
    return root;
  }

  function applyStoredPosition(root){
    const pos = safeJsonParse(localStorage.getItem(STORAGE_KEY));
    if (!pos || !Number.isFinite(pos.x) || !Number.isFinite(pos.y)) return;
    const x = clamp(pos.x, 8, window.innerWidth - Math.max(root.offsetWidth, 96) - 8);
    const y = clamp(pos.y, 8, window.innerHeight - Math.max(root.offsetHeight, 136) - 8);
    root.style.left = `${x}px`;
    root.style.top = `${y}px`;
    root.style.right = "auto";
    root.style.bottom = "auto";
  }

  function installDrag(root, orb){
    let startX = 0, startY = 0, rootX = 0, rootY = 0, moved = false, pointerId = null, dragMode = "";
    function pointFromEvent(ev){
      const touch = ev.touches?.[0] || ev.changedTouches?.[0];
      return {x: touch ? touch.clientX : ev.clientX, y: touch ? touch.clientY : ev.clientY};
    }
    function beginDrag(ev, mode){
      if(ev.button !== undefined && ev.button !== 0) return;
      if(dragMode) return;
      const pt = pointFromEvent(ev);
      dragMode = mode;
      pointerId = mode === "pointer" ? ev.pointerId : null;
      moved = false;
      startX = pt.x;
      startY = pt.y;
      const box = root.getBoundingClientRect();
      rootX = box.left;
      rootY = box.top;
      if(mode === "pointer") orb.setPointerCapture?.(ev.pointerId);
      ev.preventDefault?.();
    }
    function moveDrag(ev, mode){
      if(dragMode !== mode) return;
      if(mode === "pointer" && pointerId !== ev.pointerId) return;
      const pt = pointFromEvent(ev);
      const dx = pt.x - startX;
      const dy = pt.y - startY;
      if(Math.abs(dx) + Math.abs(dy) > 6) moved = true;
      if(!moved) return;
      const x = clamp(rootX + dx, 8, window.innerWidth - root.offsetWidth - 8);
      const y = clamp(rootY + dy, 8, window.innerHeight - root.offsetHeight - 8);
      root.style.left = `${x}px`;
      root.style.top = `${y}px`;
      root.style.right = "auto";
      root.style.bottom = "auto";
      ev.preventDefault?.();
    }
    function finishDrag(ev, mode){
      if(dragMode !== mode) return;
      if(mode === "pointer" && pointerId !== ev.pointerId) return;
      if(mode === "pointer") orb.releasePointerCapture?.(ev.pointerId);
      pointerId = null;
      dragMode = "";
      if(moved){
        const box = root.getBoundingClientRect();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({x: box.left, y: box.top}));
        setTimeout(()=>{ moved = false; }, 0);
      }
    }
    orb.addEventListener("pointerdown", ev => beginDrag(ev, "pointer"));
    orb.addEventListener("pointermove", ev => moveDrag(ev, "pointer"));
    orb.addEventListener("pointerup", ev => finishDrag(ev, "pointer"));
    orb.addEventListener("pointercancel", ev => finishDrag(ev, "pointer"));
    orb.addEventListener("mousedown", ev => beginDrag(ev, "mouse"));
    window.addEventListener("mousemove", ev => moveDrag(ev, "mouse"));
    window.addEventListener("mouseup", ev => finishDrag(ev, "mouse"));
    orb.addEventListener("touchstart", ev => beginDrag(ev, "touch"), {passive:false});
    window.addEventListener("touchmove", ev => moveDrag(ev, "touch"), {passive:false});
    window.addEventListener("touchend", ev => finishDrag(ev, "touch"));
    window.addEventListener("touchcancel", ev => finishDrag(ev, "touch"));
    orb.addEventListener("click", (ev)=>{
      if(moved){
        ev.preventDefault();
        ev.stopPropagation();
        moved = false;
        return;
      }
      root.classList.toggle("open");
    });
  }

  async function postAction(command){
    const res = await fetch("/api/assistant/actions/execute", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({instruction: command, dry_run: true, source: "xiaoyi_sprite"})
    });
    if(!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  function openFullQuestion(command){
    const url = new URL("/ops-copilot", window.location.origin);
    if(command) {
      url.searchParams.set("q", command);
      url.searchParams.set("prefill", "1");
    }
    window.location.assign(url.toString());
  }

  function isSailingAction(data){
    const id = data?.will_execute?.action_id || data?.action?.id || "";
    return data?.action?.linked_system === "sailing_simulator"
      || ["open_sailing_simulator", "start_navigation_demo", "switch_ship_view", "run_sailing_rl_smoke_test"].includes(id);
  }

  function isHubAction(data){
    const id = data?.will_execute?.action_id || data?.action?.id || "";
    return isSailingAction(data)
      || data?.action?.linked_system === "xiaoyi_ai"
      || id === "start_xiaoyi_ai";
  }

  function openIntegrationHub(command, data){
    const id = data?.will_execute?.action_id || data?.action?.id || "";
    const url = new URL("/integration-hub", window.location.origin);
    if(id) url.searchParams.set("action", id);
    if(command) url.searchParams.set("command", command);
    url.searchParams.set("auto", "1");
    window.location.assign(url.toString());
  }

  async function handleCommand(root){
    const input = root.querySelector(".xiaoyi-sprite-input");
    const submit = root.querySelector(".xiaoyi-sprite-btn[type='submit']");
    const log = root.querySelector(".xiaoyi-sprite-log");
    const command = (input?.value || "").trim();
    if(!command){
      log.textContent = "你可以直接说：小懿，开始训练碳排最低目标。";
      return;
    }
    submit.disabled = true;
    log.textContent = "收到，我先理解命令并检查是否需要进入人工确认...";
    try{
      if(isRlTrainingCommand(command)){
        const data = await postAction(command);
        if((data?.will_execute?.action_id || data?.action?.id) === "start_rl_training"){
          log.textContent = actionText(data) + "\n我会回到首页，从顶部菜单进入强化学习面板，然后弹出确认框。";
          window.setTimeout(()=>window.location.assign(guidedHomeUrl(data)), 1200);
          return;
        }
      }
      const data = await postAction(command);
      const openUrl = data?.will_execute?.open_url;
      if(isHubAction(data)){
        log.textContent = `已匹配动作：${data?.will_execute?.action_label || data?.action?.label || "项目联动动作"}。\n我会打开“项目联动中枢”，先展示接口、按钮和风险，再由你确认是否执行。`;
        window.setTimeout(()=>openIntegrationHub(command, data), 900);
        return;
      }
      if(openUrl && data?.matched){
        log.textContent = `已匹配动作：${data?.will_execute?.action_label || data?.action?.label || "可执行动作"}。\n正在打开对应界面，仍保留人工确认边界。`;
        window.setTimeout(()=>window.location.assign(openUrl), 900);
      }else{
        log.textContent = "这更像一个问答/SOP 请求，我会打开完整问答工作台；页面会等待你点击生成。";
        window.setTimeout(()=>openFullQuestion(command), 900);
      }
    }catch(err){
      log.textContent = "命令网关暂时不可用，我先打开完整问答工作台，等待你确认后再生成。";
      window.setTimeout(()=>openFullQuestion(command), 900);
    }finally{
      window.setTimeout(()=>{ submit.disabled = false; }, 1200);
    }
  }

  function install(){
    if(!document.body) return;
    createStyle();
    const root = buildWidget();
    const orb = root.querySelector(".xiaoyi-sprite-orb");
    const form = root.querySelector(".xiaoyi-sprite-form");
    applyStoredPosition(root);
    installDrag(root, orb);
    root.querySelector(".xiaoyi-sprite-close")?.addEventListener("click", ()=>root.classList.remove("open"));
    root.querySelector("[data-open-full]")?.addEventListener("click", ()=>{
      const command = (root.querySelector(".xiaoyi-sprite-input")?.value || "").trim();
      openFullQuestion(command);
    });
    form?.addEventListener("submit", (ev)=>{
      ev.preventDefault();
      handleCommand(root);
    });
    window.addEventListener("resize", ()=>{
      const box = root.getBoundingClientRect();
      const x = clamp(box.left, 8, window.innerWidth - root.offsetWidth - 8);
      const y = clamp(box.top, 8, window.innerHeight - root.offsetHeight - 8);
      root.style.left = `${x}px`;
      root.style.top = `${y}px`;
      root.style.right = "auto";
      root.style.bottom = "auto";
    });
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", install);
  }else{
    install();
  }
})();
