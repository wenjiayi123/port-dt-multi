/* Shared schematic equipment renderer. Motion illustrates mechanisms, never measured tracks. */
export async function createSpatial(host, kind, onSelect, attempt=0, onDemand=false) {
  const T = await import('/static/vendor/three/three.module.min.js?v=160.1'+(attempt?'&retry='+attempt:''));
  const scene = new T.Scene(), group = new T.Group(); scene.add(group);
  const renderer = new T.WebGLRenderer({ antialias:true, alpha:true });
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5)); renderer.outputColorSpace=T.SRGBColorSpace;
  renderer.setClearColor(0x0c1c29,0); host.replaceChildren(renderer.domElement);
  const camera=new T.PerspectiveCamera(36,1,.1,250), target=new T.Vector3(0,2,0);
  const ambient=new T.HemisphereLight(0xc5f4ff,0x273a42,2.5);scene.add(ambient);
  const sun=new T.DirectionalLight(0xffe8c1,3.2); sun.position.set(-15,30,15);scene.add(sun);
  const rim=new T.DirectionalLight(0x73d6ed,2);rim.position.set(12,10,-12);scene.add(rim);
  const mats=new Map(), selectable=[], resources=new Set(); let disposed=false,paused=matchMedia('(prefers-reduced-motion: reduce)').matches,frame=0;
  const material=c=>{if(!mats.has(c))mats.set(c,new T.MeshStandardMaterial({color:c,roughness:.65,metalness:.2}));return mats.get(c);};
  function mesh(geometry,color,x,y,z,name,parent=group){ const m=new T.Mesh(geometry,material(color));m.position.set(x,y,z);parent.add(m);resources.add(geometry); if(name){m.userData.label=name;selectable.push(m);}return m; }
  function box(x,y,z,w,h,d,c,n,parent=group){return mesh(new T.BoxGeometry(w,h,d),c,x,y,z,n,parent);}
  function cylinder(x,y,z,r,h,c,n,parent=group){return mesh(new T.CylinderGeometry(r,r,h,24),c,x,y,z,n,parent);}
  function pipe(points,c,r=.09){const path=new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)),false,'centripetal'); return mesh(new T.TubeGeometry(path,36,r,8,false),c,0,0,0);}
  function label(x,y,z,text){const cv=document.createElement('canvas');cv.width=512;cv.height=64;const ctx=cv.getContext('2d');ctx.fillStyle='#102a36';ctx.fillRect(0,0,512,64);ctx.font='28px system-ui';ctx.fillStyle='#b9e6e3';ctx.textAlign='center';ctx.fillText(text,256,43);const texture=new T.CanvasTexture(cv);resources.add(texture);const mat=new T.SpriteMaterial({map:texture,transparent:true,depthTest:false});resources.add(mat);const s=new T.Sprite(mat);s.position.set(x,y,z);s.scale.set(5.5,.7,1);group.add(s);return s;}
  function container(x,y,z,c=0x476d7b){const b=box(x,y,z,2.8,1.1,1.2,c,'集装箱堆位 · 布局示意');for(let j=0;j<8;j++)box(x-1.2+j*.34,y,z+.611,.04,.88,.03,0x78909a);return b;}
  box(0,-.35,0,27,.6,19,0x233b46,'设施基础 / 平面布置示意');
  const grid=new T.GridHelper(26,26,0x416a72,0x2a4651);grid.position.y=-.03;group.add(grid);resources.add(grid.geometry);resources.add(grid.material);
  const fans=[],batteryLevels=[]; let craneTrolley=null;
  if(kind==='hvac'){
    for(let i=0;i<3;i++){
      const x=-7+i*6;
      box(x,.4,0,4.8,.7,4.1,0x4b6670,'冷机基础');
      const drum=cylinder(x,2.1,0,1.25,4,0x88a8ae,'离心式冷水机组 · 点选查看模型设定');drum.rotation.z=Math.PI/2;
      box(x,3.3,.1,1.6,1.4,1.5,0x467b7c,'压缩机');box(x+1.65,2.8,1.1,.7,1.8,.6,0xabc0bf,'冷机控制柜');
      box(x,1.9,-5.4,3.2,3.8,2.6,0x557780,'冷却塔 · 机构示意');
      const fan=new T.Group();fan.position.set(x,3.9,-5.4);group.add(fan);for(let k=0;k<4;k++){const blade=box(0,0,0,1.9,.08,.35,0x93b4ba,'冷却塔风机',fan);blade.rotation.y=k*Math.PI/2;}fans.push(fan);
      pipe([[x-1.4,1.4,1.8],[x-1.4,1.4,4.8],[x,1.4,4.8]],0x69c8db,.13);
      pipe([[x+1.4,1.7,-1.8],[x+1.4,1.7,-3.4],[x,3,-4.5]],0xe5a06c,.12);
    }
    pipe([[-9,1,5],[0,1,5],[9,1,5]],0x69c8db,.18);pipe([[-9,1.2,-3],[0,1.2,-3],[9,1.2,-3]],0xdd966a,.16);
    label(-5,5.5,0,'冷水机组 CHILLER');label(5,5.4,-5,'冷却塔 COOLING');label(0,.8,7,'供水 / 回水流向示意');
  } else if(kind==='bess'||kind==='shore'){
    for(let i=0;i<6;i++){
      const x=-8+(i%3)*3.6,z=-3+Math.floor(i/3)*4;
      box(x,2,z,2.4,4,2.4,0x96b3b7,'储能电池柜 · 图形数量为示意');box(x,2,z+1.22,1.95,3.5,.05,0x254b56,'电池舱');
      for(let j=0;j<6;j++)batteryLevels.push({m:box(x, .6+j*.55,z+1.28,1.6,.32,.08,0x294c57,'SoC 显示 · 等待模型数据'),level:(j+1)/6});
      for(let j=0;j<4;j++)box(x+1.22,1+j*.7,z,.03,.12,1.6,0x426773);
    }
    box(4,1.6,0,3,3.2,2.3,0x5d8b98,'PCS 双向变流器');box(4,2.3,1.2,1,.65,.08,0x8cdbcc,'PCS 控制屏');
    box(8,1.5,-4,3.2,3,2.8,0x7b8f8f,'并网变压器 / PCC');
    for(let j=0;j<3;j++)cylinder(7+j,3.35,-4,.18,.8,0xc8bbae,'高压套管');
    pipe([[-9,.3,3],[0,.3,3],[4,.3,3],[4,1,1]],0x66d1c1,.12);pipe([[5.5,1,0],[8,1,0],[8,1,-3]],0xe6b96e,.13);
    label(-5,5.4,-2,'BESS 电池舱');label(5,4.7,0,'PCS / PCC');
    if(kind==='shore'){
      box(0,-.1,7,27,.15,4,0x204d62,'港池水面 · 示意');box(6,.8,7,9,1.6,2.2,0x476876,'船舶岸电连接示意');box(8,2.1,7,2,1.2,1.6,0xc7d6d5,'船舶上层建筑');pipe([[8,1,-2],[10,2,2],[9,2,6]],0x6fdac6,.13);label(4,3.6,7,'岸电连接 / SHORE POWER');
    }
  }else{
    for(let r=0;r<3;r++)for(let c=0;c<4;c++)for(let h=0;h<(c%2?2:3);h++)container(-7+c*4, .65+h*1.13,-4+r*3.1,[0x567c89,0x3c777e,0xa17b56,0x677e95][(r+c)%4]);
    if(kind==='lighting'){
      ambient.intensity=.95;sun.intensity=.65;rim.intensity=.9;
      for(const [x,z]of[[-11,-6],[10,-6],[-11,6],[10,6]]){
        cylinder(x,4.8,z,.1,9.6,0x94acb4,'高杆灯 · 位置示意');cylinder(x,9.6,z,.55,.3,0xd5ca94,'灯盘');
        const pool=new T.SpotLight(0xe5f0bc,160,22,Math.PI/5,.7,1);pool.position.set(x,9.4,z);pool.target.position.set(x*.7,0,z*.7);scene.add(pool,pool.target);
        const glow=mesh(new T.ConeGeometry(3.7,8,24,1,true),0x77ac9f,x,5.2,z);glow.material=new T.MeshBasicMaterial({color:0xc6edbd,transparent:true,opacity:.075,side:T.DoubleSide,depthWrite:false});resources.add(glow.material);
      }label(0,6,0,'分区调光 / 照度约束');
    }else{
      for(const x of[-5,5]){box(x,.03,0,.25,.08,17,0x9baeb2,'轨道');box(x,4.5,0,.45,9,.6,0xdfb16f,'场桥门腿');box(x,4.5,5,.45,9,.6,0xdfb16f,'场桥门腿');}
      box(0,9,0,11,.7,1.1,0xe0b876,'场桥横梁');box(0,9,5,11,.7,1.1,0xe0b876,'场桥横梁');
      craneTrolley=new T.Group();craneTrolley.position.set(0,8.5,2.5);group.add(craneTrolley);
      box(0,0,0,2,.5,4.7,0x81979c,'起升小车',craneTrolley);for(const x of[-.7,.7])box(x,-1.8,0,.035,3.4,.035,0xc1d1d1,'起升钢丝绳',craneTrolley);box(0,-3.5,0,2.8,.2,1.25,0xe0b36c,'吊具 / spreader',craneTrolley);
      label(0,11,2,'轨道式集装箱门式起重机');
      if(kind==='port'){
        box(0,-.05,9,30,.12,5,0x234e64,'港池水面示意');box(-4,.65,9.8,12,1.5,2.2,0x557b89,'靠泊船舶示意');for(let i=0;i<4;i++)container(-8+i*2.5,2,9.7,0xaa906b);label(6,3,8,'泊位 / QUAYSIDE');
      }
    }
  }
  let angle=.62,elevation=.48,distance=34,top=false,drag=null,lastDraw=0;
  function position(){camera.position.set(target.x+distance*Math.cos(elevation)*Math.sin(angle),target.y+distance*Math.sin(elevation),target.z+distance*Math.cos(elevation)*Math.cos(angle));camera.lookAt(target);}
  function resize(){if(disposed)return;const r=host.getBoundingClientRect();if(r.width<1||r.height<1)return;renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();position();renderer.render(scene,camera);}
  const ro=new ResizeObserver(resize);ro.observe(host);
  const events=[];function listen(type,fn,opts){host.addEventListener(type,fn,opts);events.push([type,fn,opts]);}
  const ray=new T.Raycaster(),pointer=new T.Vector2();
  listen('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY};host.setPointerCapture(e.pointerId);});
  listen('pointermove',e=>{if(!drag)return;angle-=(e.clientX-drag.x)*.008;elevation=Math.max(.13,Math.min(1.5,elevation+(e.clientY-drag.y)*.006));drag.x=e.clientX;drag.y=e.clientY;position();renderer.render(scene,camera);});
  listen('pointerup',e=>{if(!drag)return;const click=Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5;drag=null;if(!click)return;const r=host.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(selectable)[0];if(hit)onSelect(hit.object.userData.label);});
  listen('pointercancel',()=>{drag=null;});
  listen('wheel',e=>{e.preventDefault();distance=Math.max(18,Math.min(68,distance+e.deltaY*.025));position();renderer.render(scene,camera);},{passive:false});
  listen('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')angle-=.12;if(e.key==='ArrowRight')angle+=.12;if(e.key==='ArrowUp')elevation=Math.min(1.5,elevation+.1);if(e.key==='ArrowDown')elevation=Math.max(.13,elevation-.1);position();renderer.render(scene,camera);});
  function loop(now){if(disposed)return;frame=requestAnimationFrame(loop);if(document.hidden||document.body.classList.contains('simulation-active')||document.querySelector('.ppd-dialog[open]')||!host.getClientRects().length||now-lastDraw<42)return;lastDraw=now;if(!paused){for(const f of fans)f.rotation.y+=.065;if(craneTrolley)craneTrolley.position.x=Math.sin(now*.0002)*2.8;}renderer.render(scene,camera);}
  resize();if(!onDemand)frame=requestAnimationFrame(loop);
  return { reset(){angle=.62;elevation=.48;distance=36;top=false;resize();},top(){top=!top;elevation=top?1.48:.48;resize();return top;},pause(){paused=!paused;return paused;},updateSoc(value){if(!Number.isFinite(value))return;for(const {m,level}of batteryLevels){m.material=material(level<=value?0x7bddbc:0x294c57);m.userData.label='模型 SoC '+(value*100).toFixed(1)+'% · 非现场回读';}if(onDemand)renderer.render(scene,camera);},dispose(){disposed=true;cancelAnimationFrame(frame);ro.disconnect();for(const e of events)host.removeEventListener(...e);for(const r of resources)r.dispose?.();for(const m of mats.values())m.dispose();renderer.dispose();renderer.forceContextLoss();host.replaceChildren();}};
}
