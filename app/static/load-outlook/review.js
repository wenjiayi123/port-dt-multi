const $=id=>document.getElementById(id);
const sites={shanghai:"上海",los_angeles:"洛杉矶",funchal:"丰沙尔"};
const parts={test:"较早测试段",forward:"最后时间段"};
const methods={local:"本港训练模型",recent_day:"最近可用同小时",profile:"过去七天同小时中位数",native_ar:"原生自回归",last:"持续值",recent_week:"上周同小时"};
let index, rows, revision=0;
const num=v=>v.toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2});
function options(id,items){$(id).replaceChildren(...items.map(([key,value])=>new Option(value,key)));}
async function checked(file,expected){
  const response=await fetch("./"+file,{cache:"no-store"});if(!response.ok)throw Error("缺少预测回执");
  const bytes=await response.arrayBuffer();
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),x=>x.toString(16).padStart(2,"0")).join("");
  if(digest!==expected)throw Error("回执指纹不符");return JSON.parse(new TextDecoder().decode(bytes));
}
function partOptions(){options("part",Object.keys(index.sites[$("site").value]).map(k=>[k,parts[k]]));$("part").value="forward";}
async function load(){
  const n=++revision,site=$("site").value,part=$("part").value,summary=index.sites[site][part];
  $("content").hidden=true;$("status").textContent="正在验证冻结回执…";$("day").disabled=true;
  try{
    const records=await checked(site+"-"+part+".json",summary.sha);
    if(n!==revision)return;rows=records;
    options("day",records.map(r=>[r.day,r.day]));$("day").disabled=false;
    $("who").textContent=sites[site]+" · "+(site==="funchal"?"公开实测总有功功率":"公开数据校准的工程负荷")+" · "+methods[summary.point_method]+" · 对照为"+methods[summary.reference];
    $("verdict").className=summary.qualified?"strong":"hold";
    $("verdict").textContent=summary.qualified?"本分区通过回顾性业务门槛":summary.point_method===summary.reference?"本分区沿用简单基准":"本分区尚未达到晋升门槛";
    $("performance").textContent=summary.days+" 个完整评测日。平均绝对误差 "+num(summary.mae_kw)+" kW，对照 "+num(summary.reference_mae_kw)+" kW。日峰值绝对误差 "+num(summary.peak_mae_kw)+" kW，对照 "+num(summary.reference_peak_mae_kw)+" kW。区间覆盖率 "+num(100*summary.coverage)+"%。";
    $("transfer").textContent=summary.zero_shot_mae==null?"该区间校准方法在查看首轮结果后修订，本页展示历史逐日回放。":"上海权重直接迁移至本区段，平均绝对误差 "+num(summary.zero_shot_mae)+" kW。";
    $("status").textContent="冻结回执已核验。";$("content").hidden=false;draw();
  }catch(e){if(n===revision)$("status").textContent=e.message;}
}
function draw(){
  const r=rows[$("day").selectedIndex],h=r.hours;
  $("boundary").textContent="签发 "+r.issued_at+"，最新输入小时起点 "+r.last_input_hour_start+"，区间使用 "+r.feedback_dates.length+" 个已完成反馈日。";
  $("src").textContent="训练权重 "+(r.weight_sha256||"本日采用无训练权重的同小时规则")+"。";
  $("hours").replaceChildren(...h.map((v,i)=>{const tr=document.createElement("tr");for(const s of [v.utc.slice(11,16),...([v.p10,v.p50,v.p90,r.actual[i],r.baseline[i]].map(num))]){const td=document.createElement("td");td.textContent=s;tr.append(td);}return tr;}));
  const lo=Math.min(...h.map(v=>v.p10),...r.actual),hi=Math.max(...h.map(v=>v.p90),...r.actual),x=i=>46+i*662/23,y=n=>145-131*(n-lo)/Math.max(1,hi-lo);
  const points=v=>v.map((n,i)=>x(i)+","+y(n));const a=points(h.map(v=>v.p10)),b=points(h.map(v=>v.p90));
  $("plot").replaceChildren();
  function shape(tag,attrs,text){const n=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;$("plot").append(n);}
  shape("polygon",{points:a.concat(b.reverse()).join(" "),fill:"#c5e5da"});
  shape("polyline",{points:points(h.map(v=>v.p50)).join(" "),fill:"none",stroke:"#0c7067","stroke-width":3});
  shape("polyline",{points:points(r.actual).join(" "),fill:"none",stroke:"#9b5526","stroke-width":2.4});
  for(const [yy,val] of [[16,hi],[148,lo]])shape("text",{x:1,y:yy,"font-size":11},Math.round(val)+"");
  for(const i of [0,6,12,18,23])shape("text",{x:x(i)-8,y:169,"font-size":12},String(i).padStart(2,"0"));
}
try{
  if(location.pathname.startsWith("/static/")){$("back").href="/#twin3d-section";$("back").textContent="返回 TwinLab 数字孪生";}
  index=await checked("index.json","1b16693fc87ee0f170b45d643250bfc291b918fe35c664d87014d0451ef5d66d");options("site",Object.entries(sites));partOptions();await load();
  $("site").onchange=()=>{partOptions();load()};$("part").onchange=load;$("day").onchange=draw;
}catch(e){$("content").hidden=true;$("status").textContent=e.message;}
