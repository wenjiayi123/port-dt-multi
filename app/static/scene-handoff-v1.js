const byId = id => document.getElementById(id);
const methods = ["selected", "idle", "best_fixed"];
const requestType = "gullmate.shore.scene.v1";
const receiptType = "gullmate.shore.return.v1";
const receiptHashes = new WeakMap();

function serviceUrl(port, path) {
  const url = new URL(path, location.origin);
  url.port = port;
  return url;
}

function cursor() {
  return {method: byId("method").value, hour: Number(byId("hour").value)};
}

function validCursor(value) {
  return methods.includes(value.method) && Number.isInteger(value.hour) && value.hour >= 0 && value.hour < 168;
}

function sorted(value) {
  if (Array.isArray(value)) return value.map(sorted);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, sorted(value[key])]));
  }
  return value;
}

function checksum(result) {
  if (!receiptHashes.has(result)) {
    const fields = ["schema", "part", "week", "hours", "weight_sha256", "runtime_sha256",
      "production_authority", "methods", "frames", "carbon_delta_by_component"];
    const body = fields.map(key => sorted(result[key]));
    const bytes = new TextEncoder().encode(JSON.stringify(body));
    receiptHashes.set(result, crypto.subtle.digest("SHA-256", bytes).then(buf =>
      Array.from(new Uint8Array(buf), b => b.toString(16).padStart(2, "0")).join("")));
  }
  return receiptHashes.get(result);
}

export function installHandoff(frozen, getResult, drawSeries) {
  const launch = byId("bindScene"), status = byId("bindStatus");
  const twin = serviceUrl("8000", "/static/shore-policy-scene.html");
  const main = serviceUrl("8010", "/web/policy-evidence/shore-twin/index.html");
  const q = new URLSearchParams(location.search);
  const inbound = {id:q.get("handoff"), week:q.get("week"), method:q.get("method"), hour:Number(q.get("hour"))};
  const linked = !frozen && location.origin===twin.origin && !!window.opener && q.has("hour") &&
    /^[0-9a-f-]{36}$/.test(inbound.id) && /^\d{4}-\d{2}-\d{2}$/.test(inbound.week) && validCursor(inbound);
  let pending = null, firstLoad = linked, sent = null;
  const show = msg => { status.textContent = msg; };
  function select(value) {
    byId("method").value=value.method;
    byId("hour").value=String(value.hour);
    drawSeries();
  }
  function sync() {
    const result=getResult();
    launch.hidden=!frozen && !linked;
    launch.disabled=!result || (!frozen && result.week!==inbound.week);
    if (linked && firstLoad && result?.week===inbound.week) {
      firstLoad=false;
      select(inbound);
      show("已载入小鸥选定的周、策略与小时 / Window restored from GullMate.");
    }
  }
  if (frozen) {
    launch.textContent="在 TwinLab 重算本周 / Recompute week in TwinLab";
    launch.addEventListener("click", () => {
      const result=getResult();
      if (!result) return;
      const id=crypto.randomUUID(), position=cursor(), url=new URL(twin);
      url.search=new URLSearchParams({handoff:id,week:result.week,...position});
      const child=window.open(url.toString(),"_blank");
      if (!child) {show("浏览器未打开孪生窗口，请允许弹窗后重试。");return;}
      pending={id,child,result,position};
      show("等待 TwinLab 本周重算 / Waiting for this week’s native replay.");
    });
    window.addEventListener("message", async e => {
      const value=e.data, active=pending;
      if (!active || e.origin!==twin.origin || e.source!==active.child || value?.type!==receiptType || value.id!==active.id) return;
      const reply=ok=>e.source.postMessage({type:requestType,id:active.id,ok},twin.origin);
      const stillAtOrigin=()=>getResult()===active.result && cursor().hour===active.position.hour && cursor().method===active.position.method;
      if (!stillAtOrigin()) {show("小鸥已切换业务窗口，本次回传未应用。");reply(false);return;}
      const digest=await checksum(active.result);
      if (pending!==active || !stillAtOrigin() || value.week!==active.result.week || !validCursor(value) || digest!==value.digest) {
        show("周、位置或回执校验失败 / Return could not be verified.");reply(false);return;
      }
      pending=null;
      select(value);
      show("已核对本周三种策略的 504 个小时记录，返回第 "+(value.hour+1)+" 小时 / Returned to verified hour "+(value.hour+1)+" · "+value.id);
      reply(true);
    });
  } else if (linked) {
    launch.textContent="带回此小时至小鸥 / Return this hour to GullMate";
    launch.addEventListener("click", async () => {
      const result=getResult(), position=cursor();
      if (!result || result.week!==inbound.week || !validCursor(position) || !window.opener || window.opener.closed) {
        show("原窗口不可用 / Origin window unavailable.");return;
      }
      const digest=await checksum(result);
      if (result!==getResult()) return;
      sent=inbound.id;
      window.opener.postMessage({type:receiptType,id:sent,week:result.week,...position,digest},main.origin);
      show("已发送，等待原窗口核对 / Awaiting confirmation.");
    });
    window.addEventListener("message", e => {
      if (!sent || e.origin!==main.origin || e.source!==window.opener || e.data?.type!==requestType || e.data.id!==sent) return;
      sent=null;
      if (e.data.ok===true) window.close();
      else show("小鸥未接收，请从当前业务窗口重新打开 / Return was declined.");
    });
  }
  sync();
  if (linked) {
    byId("date").value=inbound.week;
    if (byId("date").value===inbound.week) queueMicrotask(()=>byId("request").requestSubmit());
  }
  return sync;
}
