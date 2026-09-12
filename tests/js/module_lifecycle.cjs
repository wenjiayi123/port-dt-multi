const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').resolve(__dirname, '../../app/ui/index.html'),'utf8');
const slice=(start,end)=>source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)));
const settle=async()=>{for(let i=0;i<12;i++) await Promise.resolve();};
function harness(){
  const timers=new Map(),frames=new Map(),events={},pending=[];
  let next=0;
  const ctx={Promise,console,Date,Math,innerHeight:800,visible:true,fetches:0,draws:0,hours:[],posts:0,
    document:{hidden:false,getElementById:()=>null,addEventListener:(name,cb)=>{events[name]=cb;}},
    window:{addEventListener:(name,cb)=>{events[name]=cb;}},
    requestAnimationFrame:cb=>{const id=++next;frames.set(id,cb);return id;},
    cancelAnimationFrame:id=>frames.delete(id),
    setTimeout:cb=>{const id=++next;timers.set(id,cb);return id;},
    clearTimeout:id=>timers.delete(id),
    setInterval:cb=>{const id=++next;timers.set(id,cb);return id;},
    clearInterval:id=>timers.delete(id)};
  ctx.cv={getClientRects:()=>ctx.visible?[{}]:[],getBoundingClientRect:()=>({top:0,bottom:400})};
  ctx.fetch=()=>{ctx.fetches++;return new Promise(resolve=>pending.push(resolve));};
  vm.createContext(ctx);
  return {ctx,timers,frames,events,pending};
}
async function testPortviz(){
  const h=harness(),{ctx,timers,frames,pending}=h;
  vm.runInContext(`let running=true,portvizVisible=true,frameRequest=0,lastTs=0,speed=1; function frame(){} function applyFrame(){} function resize(){}\n${slice('  function canRunPortviz(){','  // —— 单体绘制')}\n${slice('  let portvizPullTimer = null;','  function applyFrame(f)')}\nthis.api={sync:syncPortvizModule,pause:()=>{running=false;syncPortvizActivity();}}`,ctx);
  ctx.api.sync();ctx.api.sync();
  assert.equal(ctx.fetches,1,'repeated show must coalesce stream requests');
  assert.equal(frames.size,1);
  ctx.visible=false;ctx.api.sync();
  assert.equal(frames.size,0,'leaving cancels rendering synchronously');
  pending.shift()({ok:true,json:async()=>({ts:5})});await settle();
  assert.equal(timers.size,0,'hidden request completion must not restart polling');
  ctx.visible=true;ctx.api.sync();
  assert.equal(ctx.fetches,2,'returning requests a fresh frame');
  pending.shift()({ok:true,json:async()=>({ts:6})});await settle();
  assert.equal(timers.size,1,'only one timer can resume');
  ctx.visible=false;ctx.api.sync();
  assert.equal(timers.size,0,'leaving clears scheduled polling');
  ctx.api.pause();ctx.visible=true;ctx.api.sync();
  assert.equal(ctx.fetches,2,'user pause remains in effect after return');
  assert.equal(frames.size,0);
}
async function testStory(){
  const {ctx,timers,events}=harness();
  const btn={};let finishPost;
  ctx.$=id=>id==='story-section'?ctx.cv:btn;
  ctx.setHour=h=>ctx.hours.push(h);
  ctx.postPlay=()=>{ctx.posts++;return new Promise(resolve=>{finishPost=resolve;});};
  vm.runInContext(`let __storyAvailable=true;
${slice('    let __storyTimer = null;','    // —— 初始化绑定 ——')}\nthis.play=handlePlay;`,ctx);
  const started=ctx.play();
  assert.equal(ctx.posts,1);
  ctx.visible=false;events['port-module-hidden']();finishPost();await started;
  assert.equal(timers.size,0,'slow POST response cannot revive closed Story');
  assert.equal(ctx.hours.length,0,'closed Story cannot advance');
  assert.equal(btn.textContent,'▶ 继续播放');
  ctx.visible=true;await ctx.play();
  assert.equal(ctx.posts,1,'resume does not restart the backend story');
  assert.equal(timers.size,1);
  assert.equal(ctx.hours.at(-1),-24);
  timers.values().next().value();
  assert.equal(ctx.hours.at(-1),-23);
  ctx.visible=false;events['port-module-hidden']();
  assert.equal(timers.size,0);
  ctx.visible=true;await ctx.play();
  assert.equal(ctx.hours.at(-1),-23,'resume keeps the prior hour');
  assert.equal(timers.size,1);
  await ctx.play();assert.equal(timers.size,0,'manual pause remains working');
}
async function testThree(){
  const {ctx,frames,events}=harness();
  ctx.host=ctx.cv;ctx.scene={};ctx.camera={};ctx.renderer={render:()=>ctx.draws++};ctx.controls={};ctx.entities=[];ctx.three={};
  ctx.twinSceneVisible=()=>ctx.visible&&!ctx.document.hidden;
  vm.runInContext(slice('      three={scene,camera,renderer,controls,entities,anim:0};','      enablePicking(); refresh3DColors(); setInterval'),ctx);
  assert.equal(frames.size,1);
  const [id,cb]=frames.entries().next().value;frames.delete(id);cb();
  assert.equal(ctx.draws,1);assert.equal(frames.size,1);
  ctx.visible=false;events['port-module-hidden']();
  assert.equal(frames.size,0,'hidden THREE view cancels render loop');
  ctx.visible=true;events['port-module-shown']();events['port-module-shown']();
  assert.equal(frames.size,1,'repeated show must not fork render loops');
  ctx.document.hidden=true;events.visibilitychange();assert.equal(frames.size,0);
}
function testV3ReturnHome(){
  const script=fs.readFileSync(require('node:path').resolve(__dirname, '../../app/ui/v3/v3.js'),'utf8');
  const handler=script.slice(script.indexOf('function returnToHome('),script.indexOf('function safeNumber('));
  for(const previous of ['/ops-copilot','/#exec-cockpit','/']){
    let destination=null,prevented=false;
    const context={window:{location:{search:'?from=home',assign:url=>{destination=url;}},history:{length:3,back:()=>{throw Error('Return home must not reopen '+previous);}}},document:{referrer:'http://localhost'+previous}};
    vm.createContext(context);vm.runInContext(handler,context);
    context.returnToHome({preventDefault:()=>{prevented=true;}});
    assert.equal(destination,'/');assert.equal(prevented,true);
  }
}
(async()=>{await testPortviz();await testStory();await testThree();testV3ReturnHome();console.log('MODULE_LIFECYCLE:PASS:portviz,story,three,v3-home');})().catch(error=>{console.error(error);process.exitCode=1;});
