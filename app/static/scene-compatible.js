/* Local SVG fallback only. Geometry is schematic; no operational values are fabricated. */
export function createCompatibleScene(host, kind = 'port', onSelect = () => {}, layout = null) {
  const ns = 'http://www.w3.org/2000/svg';
  const make = (tag, attrs = {}, text = '') => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text) node.textContent = text;
    return node;
  };
  const root = document.createElement('div');
  root.style.cssText = 'position:absolute;inset:0;background:linear-gradient(145deg,#102a38,#091824);overflow:hidden';
  const svg = make('svg', {viewBox:'0 0 800 460',role:'img','aria-label':'兼容设备示意图，非现场布局',width:'100%',height:'100%'});
  const group = make('g'); svg.append(group); root.append(svg); host.prepend(root);
  const caption = make('text',{x:26,y:30,fill:'#a1d8d4','font-size':13,'font-family':'system-ui'},'兼容示意视图 · WebGL 不可用 · 非现场布局'); svg.append(caption);
  const status = make('text',{x:26,y:438,fill:'#90b2c2','font-size':12,'font-family':'system-ui'},'点选设备查看说明；业务指标继续取原有数据接口'); svg.append(status);
  const schemes = {
    port:[['quay','前沿泊位'],['yard','集装箱堆场'],['transport','水平运输'],['energy','岸电与储能'],['gate','闸口与铁路']],
    lighting:[['lights','高杆照明'],['yard','堆位与道路'],['control','照度与调光约束']],
    hvac:[['chiller','冷机'],['tower','冷却塔'],['pump','循环水泵'],['supply','供回水管路']],
    shore:[['vessel','靠泊船舶'],['shore','岸电连接'],['bess','电池舱'],['pcs','变流器与并网点']],
    bess:[['battery','电池舱'],['pcs','变流器'],['grid','并网点']],
    crane:[['rail','轨道与堆位'],['gantry','场桥门架'],['trolley','起升小车']]
  };
  let rows = schemes[kind] || schemes.port, current = null, top = true, shade = 'day', disposed = false;
  const shapes = new Map();
  rows.forEach(([key,label], i) => {
    const col=i%3,row=Math.floor(i/3),x=48+col*249,y=98+row*143;
    const cell=make('g',{role:'button',tabindex:0,'aria-label':label+' · 结构示意','data-part':key,style:'cursor:pointer'});
    cell.append(make('path',{d:`M${x},${y} l22,-15 h171 l-22,15 z`,fill:'#3c6577'}));
    cell.append(make('rect',{x,y,width:171,height:75,rx:7,fill:'#254858',stroke:'#6db3bc','stroke-width':1.5}));
    for(let j=0;j<6;j++)cell.append(make('path',{d:`M${x+16+j*25},${y+14} v31`,stroke:'#537e8d','stroke-width':6}));
    cell.append(make('text',{x:x+85.5,y:y+61,'text-anchor':'middle',fill:'#d4ebe9','font-size':13,'font-family':'system-ui'},label));
    const select=()=>{focus(key);onSelect(key,label+' · 兼容结构示意，非现场测量');};
    cell.addEventListener('click',select);cell.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
    group.append(cell);shapes.set(key,cell);
  });
  function focus(key){current=key;shapes.forEach((node,k)=>{node.style.opacity=(!key||key===k)?'1':'.45';});status.textContent=key?(rows.find(r=>r[0]===key)?.[1]||key)+' · 结构示意；对应业务指标见原有详情':'点选设备查看说明；业务指标继续取原有数据接口';}
  host.dataset.renderer='svg-compatible';host.dataset.renderState='ready';
  if(layout)caption.textContent+=' · '+(layout.yards?.length??0)+' 个布局区块';
  return {
    focus, view(key){focus(key==='overview'?null:key);}, get focusKey(){return current||'quay';},
    reset(){top=true;group.removeAttribute('transform');focus(null);},
    top(){top=!top;group.setAttribute('transform',top?'':'translate(90 38) skewX(-12) scale(.9 .86)');return top;},
    pause(){return true;},redraw(){},layer(value){root.dataset.layer=value;focus(value==='transport'?'transport':value==='services'?'energy':null);},
    theme(value){shade=value;root.style.filter=shade==='night'?'brightness(.78)':'none';},
    update(frame){if(!disposed&&frame?.vessels)status.textContent=`回放帧含 ${frame.vessels.length} 条船舶记录 · 设备排列为示意`;},
    updateSoc(value){if(Number.isFinite(value))status.textContent=`模型 SoC ${(value*100).toFixed(1)}% · 非现场回读`;},
    dispose(){disposed=true;root.remove();delete host.dataset.renderer;delete host.dataset.renderState;}
  };
}
