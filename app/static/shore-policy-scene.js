import {installHandoff} from "./scene-handoff-v1.js";
const el = id => document.getElementById(id);
const names = {selected:"受训策略", idle:"待机对照", best_fixed:"固定优化"};
const number = new Intl.NumberFormat("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2});
const frozen = document.documentElement.dataset.source === "archive";
let frozenWeeks;
async function readVerified(path, expected) {
  const response = await fetch(path,{cache:"no-store"});
  if (!response.ok) throw Error("未取得冻结回执");
  const buffer = await response.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256",buffer);
  const actual = Array.from(new Uint8Array(digest), b=>b.toString(16).padStart(2,"0")).join("");
  if (actual !== expected) throw Error("冻结回执哈希不符");
  return JSON.parse(new TextDecoder().decode(buffer));
}
async function loadWeek(day) {
  if (frozen) {
    frozenWeeks ??= await readVerified("./weeks.json","");
    if (!Object.hasOwn(frozenWeeks,day)) throw Error("未归档该周");
    return readVerified("./"+day+".json",frozenWeeks[day]);
  }
  const response = await fetch("/api/rl/shore-storage/weekly/"+encodeURIComponent(day)+"?include_frames=true",{cache:"no-store"});
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === "string" ? data.detail : "输入日期无效");
  return data;
}
let result = null;
let scene = null;
let version = 0;
let mode = "closed";

function closeScene() {
  ++version;
  scene?.dispose();
  scene = null;
  mode = "closed";
  el("stage").replaceChildren();
  el("stage").hidden = true;
  el("top").hidden = true;
  el("toggle").textContent = "打开三维场景";
  el("toggle").setAttribute("aria-expanded","false");
  el("mode").textContent = "按需加载三维设备。切换小时或策略时，电池舱同步显示模型荷电状态。";
  el("picked").textContent = "";
}

async function openScene() {
  if (mode !== "closed") { closeScene(); return; }
  if (!result) return;
  const id = ++version;
  const host = document.createElement("div");
  host.style.cssText = "width:100%;height:100%;position:relative";
  host.tabIndex = 0;
  const pick = label => { if (id === version) el("picked").textContent = label; };
  el("stage").replaceChildren(host);
  el("stage").hidden = false;
  el("toggle").textContent = "关闭场景";
  el("toggle").setAttribute("aria-expanded","true");
  mode = "loading";
  el("mode").textContent = "正在加载本机设备模型…";
  try {
    let instance, renderer;
    try {
      const {createSpatial} = await import("/static/module-spatial.js");
      if (id !== version) return;
      instance = await createSpatial(host,"shore",pick,0,true);
      renderer = "WebGL 三维设备示意";
    } catch {
      if (id !== version) return;
      const {createCompatibleScene} = await import("/static/scene-compatible.js");
      if (id !== version) return;
      instance = createCompatibleScene(host,"shore",pick);
      renderer = "二维兼容设备示意";
    }
    if (id !== version) { instance.dispose(); return; }
    scene = instance;
    mode = renderer;
    el("top").hidden = false;
    drawHour();
  } catch (error) {
    if (id === version) {
      closeScene();
      el("mode").textContent = "设备图未加载。逐时电表仍可查阅。";
    }
  }
}

function drawHour() {
  const h = Number(el("hour").value);
  const series = result.frames[el("method").value];
  const frame = series[h];
  const timestamp = new Date(Date.parse(result.week+"T00:00:00Z") + h*3600000);
  el("time").value = "第 "+(h+1)+" 小时 · "+timestamp.toISOString().slice(0,16).replace("T"," ")+" UTC 起";
  const values = [["电网输入",frame.pcc_kw,"kW"],["电池放电为正",frame.bess_kw,"kW"],["柔性负荷变化",frame.flex_kw,"kW"],
                  ["模型保障岸电",frame.mandatory_shore_kw,"kW"],["小时末荷电状态",100*frame.soc,"%"]];
  el("metrics").replaceChildren(...values.map(([label,val,unit]) => {
    const cell = document.createElement("div"), term = document.createElement("dt"), value = document.createElement("dd");
    term.textContent = label; value.textContent = number.format(val)+" "+unit;
    cell.append(term,value); return cell;
  }));
  syncHandoff();
  const x = h*690/167+5;
  el("cursor").setAttribute("x1",x);
  el("cursor").setAttribute("x2",x);
  if (scene) {
    scene.updateSoc(frame.soc);
    el("mode").textContent = mode+" · "+names[el("method").value]+" · 第 "+(h+1)+" 小时末荷电状态 "+number.format(100*frame.soc)+"%";
  }
}

function drawSeries() {
  const series = result.frames[el("method").value];
  const max = Math.max(...Object.values(result.frames).flat().map(v => v.pcc_kw), 1);
  const path = series.map((v,i) => (i ? "L":"M")+(i*690/167+5).toFixed(2)+","+(145-140*v.pcc_kw/max).toFixed(2)).join(" ");
  el("line").setAttribute("d",path);
  el("chart").setAttribute("aria-label",names[el("method").value]+"，横轴 1 至 168 小时，纵轴 0 至 "+number.format(max)+" 千瓦");
  drawHour();
}

el("request").addEventListener("submit",async event => {
  event.preventDefault();
  if (el("run").disabled) return;
  const day = el("date").value;
  closeScene();
  result = null;
  syncHandoff();
  el("results").hidden = true;
  el("date").disabled = el("run").disabled = true;
  el("message").textContent = frozen ? "正在核对归档哈希…" : "TwinLab 正在运行三种策略…";
  try {
    const data = await loadWeek(day);
    if (data.week !== day || data.production_authority !== false || data.weight_sha256 !== "1cedb00d486f4e7cd9f318242a52c120f4b851b7545de711915cf9d899c6b1bc" ||
        !Object.keys(names).every(k=>data.frames?.[k]?.length === 168)) throw Error("孪生回执身份或逐时数据不完整");
    result = data;
    el("hour").value = 0;
    el("totals").replaceChildren(...Object.entries(names).map(([key,label]) => {
      const tr = document.createElement("tr"), row = data.methods[key];
      for (const value of [label,...["total_cost_cny","carbon_kg","peak_kw"].map(k=>number.format(row[k]))]) {
        const td = document.createElement("td"); td.textContent = value; tr.append(td);
      }
      return tr;
    }));
    el("source").textContent = (frozen?"归档时由原生 TwinLab 重算，本次已核对文件哈希":"原生 TwinLab 本次重算")+" · 策略权重 SHA-256 "+data.weight_sha256+" · 现场控制权限关闭";
    el("results").hidden = false;
    el("message").textContent = day+(frozen?" 起始周已读取冻结回执":" 起始周已重算")+"，每种策略 168 小时。";
    drawSeries();
  } catch (error) { el("message").textContent = "未取得本次结果 · "+error.message; }
  finally { el("date").disabled = el("run").disabled = false; }
});
el("date").addEventListener("change",() => {
  closeScene(); result = null; syncHandoff(); el("results").hidden = true; el("message").textContent = frozen?"日期已更改，请读取回执。":"日期已更改，请重新计算。";
});
el("method").addEventListener("change",drawSeries);
el("hour").addEventListener("input",drawHour);
el("toggle").addEventListener("click",openScene);
el("top").addEventListener("click",() => { scene?.top(); });
document.addEventListener("visibilitychange",() => { if (document.hidden) closeScene(); });
window.addEventListener("pagehide",closeScene);
const syncHandoff=installHandoff(frozen,()=>result,drawSeries);
el("run").disabled = false;
