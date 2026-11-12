const el=id=>document.getElementById(id);
const scenarios={shanghai:"上海原电价",industrial_mix:"叠加工业负荷",industrial_mix_price_shift:"工业负荷 · 电价延后 4 小时",phase_minus3:"电价提前 3 小时",phase_plus6:"电价延后 6 小时",week_switch:"周内两次切换",unlisted_tariff:"未知电价表"};
const sites={fresh_shanghai:"上海 · 新月份",funchal_same_period:"丰沙尔 · 原时段复验",all45_funchal:"Øra 全 45 户 · 丰沙尔权重",all45_shanghai:"Øra 全 45 户 · 上海权重",fixed42_funchal:"Øra 同组 42 户 · 丰沙尔权重",fixed42_shanghai:"Øra 同组 42 户 · 上海权重",industrial_mix:"上海与工业负荷混合", "all45_clock_-1":"Øra 时间偏移 −1 小时","all45_clock_+1":"Øra 时间偏移 +1 小时"};
const policy={idle:"待机",original:"原受训策略",paired_cap_margin:"碳排余量",peak_guard:"已观测峰值保护",adaptive_clock:"识别电价后的受训策略"};
const statuses={confirmed:"已区分时段",native_consistent:"读数符合原表",unconfirmed:"无法确认",missing:"读数不足"};
let mode="control", index, data, picked, rev=0;
const n=v=>Number(v).toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2});
const canPlot=v=>Array.isArray(v)&&v.length>0;
async function read(file,hash){
 const res=await fetch("./"+file,{cache:"no-store"});if(!res.ok)throw Error("无法取得归档文件");
 const b=await res.arrayBuffer();const raw=await crypto.subtle.digest("SHA-256",b);
 if(Array.from(new Uint8Array(raw),x=>x.toString(16).padStart(2,"0")).join("")!==hash)throw Error("回执指纹不符");
 return JSON.parse(new TextDecoder().decode(b));
}
function options(id,rows){el(id).replaceChildren(...rows.map(([k,v])=>new Option(v,k)));}
function row(v,tag="td"){const tr=document.createElement("tr");for(const s of v){const c=document.createElement(tag);c.textContent=s;tr.append(c);}return tr;}
function drawLines(values,other){
 const svg=el("plot");svg.replaceChildren();const min=Math.min(...values,...other),max=Math.max(...values,...other);
 function shape(tag,attrs,txt){const a=document.createElementNS("http://www.w3.org/2000/svg",tag);Object.entries(attrs).forEach(([k,v])=>a.setAttribute(k,v));a.textContent=txt||"";svg.append(a);}
 for(const [a,stroke] of [[other,"#9a562d"],[values,"#107263"]]){
   const pts=a.map((v,i)=>(48+i*642/(a.length-1)).toFixed(2)+","+(128-(v-min)*112/Math.max(1,max-min)).toFixed(2)).join(" ");
   shape("polyline",{points:pts,fill:"none",stroke,"stroke-width":2});
 }
 for(const [y,v] of [[17,max],[130,min]])shape("text",{x:2,y,"font-size":11},Math.round(v)+"");
 shape("line",{id:"cursor",y1:7,y2:136,stroke:"#173936","stroke-width":1});
}
function hour(){
 const h=Number(el("hour").value),c=mode==="control",count=c?168:24,x=48+h*642/(count-1);
 el("cursor").setAttribute("x1",x);el("cursor").setAttribute("x2",x);
 if(c){const v=picked.hours[h];el("hourVal").textContent="第 "+(h+1)+" 小时 · 原策略 "+n(v[0])+" kW，电价识别策略 "+n(v[1])+" kW，电池 "+n(v[2])+" kW，小时末 SOC "+n(v[3]*100)+"%。";}
 else {el("hourVal").textContent=picked.hour_clock+" · 第 "+(h+1)+" 小时 · 中心预测 "+n(picked.center[h])+" kW，实绩 "+n(picked.actual[h])+" kW。";}
}
function dateChanged(){
 const c=mode==="control",key=el("source").value,s=index[mode][key];
 picked=data[el("date").selectedIndex];
 el("timeline").hidden=!c;el("hour").max=c?167:23;el("hour").value=0;
 if(c){
   const d=s.policies.adaptive_clock,ci=d.vs_original.three_week_ci[0];
   el("heading").textContent=scenarios[key]+" · 17 周";
   el("scope").textContent="2026 年 6–9 月。各工况共用同一组自然周，每周用同一组初态和冻结的 RL 权重运行。";
   el("outcome").textContent="电价识别版相对原策略，平均每周节省 "+n(d.vs_original.mean[0])+" 元；相对待机节省 "+n(d.vs_idle.mean[0])+" 元。";
   el("uncertainty").textContent="与原策略相比，节省额的 95% 三周分块区间为 "+n(ci[0])+" 至 "+n(ci[1])+" 元。负值表示费用增加。";
   el("detail").textContent=picked.day+" 起始周结算";
   el("head").replaceChildren(row(["控制方式","账单（元）","碳排（kg）","峰值（kW）","较原策略节省（元）"],"th"));
   el("rows").replaceChildren(...Object.entries(picked.policies).map(([k,v])=>row([policy[k],...v.totals.map(n),n(v.vs_original[0])])));
   el("dispatchNote").textContent=key==="unlisted_tariff"?"无法确认电价表的规划段退到待机。":"原电价保持适用时沿用原控制；确认偏移后才调整优化器采用的电价。";
   el("chartTitle").textContent="电网输入曲线";el("legend").textContent="绿色为电价识别版，棕色为原策略。";
   el("plans").replaceChildren(...picked.tariff_plans.map(p=>row([p.at.slice(5,16).replace("T"," "),p.span+" h",n(p.current_price)+" 元/kWh",statuses[p.observed.status],p.phase_used===null?"待机":p.phase_used===0?"按原表":"偏移 "+(p.phase_used>12?p.phase_used-24:p.phase_used)+" h"])));
   drawLines(picked.hours.map(a=>a[1]),picked.hours.map(a=>a[0]));
   el("cmd").textContent="G:\\AI\\WindowsDeploy\\portai.cmd stress "+key+" "+picked.day;
 }else{
   el("heading").textContent=sites[key]+" · "+s.scored+" 个完整日";
   el("scope").textContent=key.startsWith("all45")||key.startsWith("fixed42")?"挪威工业配网历史记录。跨站移植在原模型训练完成后离线复算。":key==="funchal_same_period"?"同一丰沙尔采集期的回顾性修订。":"上海工程负荷，未纳入冻结模型的训练或选择。";
   el("outcome").textContent="功率平均绝对误差 "+n(s.mae)+" kW；最近可用同小时基准 "+n(s.recent)+" kW，七日同小时中位数 "+n(s.profile)+" kW。";
   el("uncertainty").textContent="日峰值误差从中心曲线取最大时为 "+n(s.center_peak)+" kW，单独估峰后为 "+n(s.day_peak)+" kW。";
   el("detail").textContent=picked.day+" 的日峰值";el("head").replaceChildren(row(["实绩 kW","独立估峰 kW","中心曲线最大 kW","以往完整日 kW"],"th"));
   el("rows").replaceChildren(row([Math.max(...picked.actual),picked.day_peak,picked.center_peak,picked.last_confirmed_peak].map(n)));
   el("dispatchNote").textContent="签发 "+picked.issued_at+"，最近反馈日 "+picked.last_feedback_day+"。日峰值估计不指定峰值出现的小时。";
   el("legend").textContent="绿色为日前预测，棕色为后来观测或工程回放的实绩。";el("chartTitle").textContent="24 小时负荷";
   el("cmd").textContent=["fresh_shanghai","funchal_same_period"].includes(key)?"G:\\AI\\WindowsDeploy\\portai.cmd forecast "+(key==="fresh_shanghai"?"shanghai":"funchal")+" "+picked.day+" --daily-peak":"G:\\AI\\WindowsDeploy\\portai.cmd stress shanghai 2026-06-01";
   drawLines(picked.center,picked.actual);
 }hour();
}
async function sourceChanged(){
 const sequence=++rev,key=el("source").value,meta=index[mode][key];el("result").hidden=true;el("date").disabled=true;el("status").textContent="正在校验回执…";
 try{const d=await read(mode+"-"+key+".json",meta.sha);if(sequence!==rev)return;data=d;options("date",d.map(v=>[v.day,v.day]));dateChanged();el("result").hidden=false;el("date").disabled=false;el("status").textContent="冻结回执已核验。";}
 catch(e){if(sequence===rev)el("status").textContent=e.message;}
}
function selectMode(v){
 mode=v;const c=v==="control";el("ctrl").setAttribute("aria-pressed",c);el("fc").setAttribute("aria-pressed",!c);
 el("sourceLabel").textContent=c?"工况":"数据与预测模型";el("dateLabel").textContent=c?"周起点（UTC）":"签发日";
 options("source",Object.entries(c?scenarios:sites));el("source").value=c?"industrial_mix_price_shift":"fresh_shanghai";return sourceChanged();
}
try{index=await read("index.json","52f010bde54361fe54fa2c9458cf0dfa205615d3f486846813750dab4a586547");await selectMode(mode);el("ctrl").onclick=()=>selectMode("control");el("fc").onclick=()=>selectMode("forecast");el("source").onchange=sourceChanged;el("date").onchange=dateChanged;el("hour").oninput=hour;}
catch(e){el("status").textContent=e.message;}
