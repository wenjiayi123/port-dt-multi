// Offline VM contracts for the actual navigation controller. Not browser evidence.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),test=require('node:test');
const main=fs.readFileSync('app/ui/index.html','utf8');
const menu=main.match(/<nav\b(?=[^>]*\bid=["']panel-nav-primary["'])[^>]*>[\s\S]*?<\/nav>/)[0].replace(/\b(href|data-direct-target)=(["'])#/g,'$1=$2/#');
const source=fs.readFileSync('app/static/standalone_navigation.js','utf8');
const cacheKey='port-dt.module-menu.v20260912-2';
const settle=async()=>{for(let i=0;i<30;i++)await Promise.resolve()};
function harness({cache='',path='/v3',responses=[200],bodyReady=true}={}){
 const windowEvents={},docEvents={},timers=new Map(),requests=[],warnings=[],routes=[],storage=new Map([[cacheKey,cache]]);let nextTimer=0,host;
 const location={href:'http://unit.test'+path,pathname:path.split('?')[0],assign:url=>routes.push(url)};
 const add=(store,name,fn)=>(store[name]??=[]).push(fn);
 const emit=(store,name,event={})=>(store[name]||[]).forEach(fn=>fn(event));
 function element(attrs={}){return {attrs,handlers:{},style:{setProperty(){}},textContent:'',offsetHeight:60,offsetWidth:20,clientWidth:800,scrollWidth:800,
  classList:{add(){},remove(){}},setAttribute(k,v){this.attrs[k]=v},removeAttribute(k){delete this.attrs[k]},getAttribute(k){return this.attrs[k]},
  addEventListener(name,fn){add(this.handlers,name,fn)},contains(){return false},focus(){},getBoundingClientRect(){return{left:0,bottom:60}},querySelector(){return null},querySelectorAll(){return []}}}
 function navFrom(markup){
  if(!markup.includes('id="panel-nav-primary"'))return null;
  const nav=element();nav.links=[...markup.matchAll(/<a\b[^>]*href="([^"]*)"[^>]*>/g)].map(m=>{
   const a=element({href:m[1]});a.href=m[1];return a;
  });
  nav.querySelector=sel=>sel==='a[href]'?nav.links[0]:sel==='[aria-current="page"]'?nav.links.find(a=>a.attrs['aria-current']==='page'):null;
  nav.querySelectorAll=sel=>sel==='a[href]'?nav.links:[];return nav;
 }
 const document={body:bodyReady?element():null,documentElement:element(),
  getElementById:id=>id==='standalone-module-navigation'?host:null,
  querySelector:selector=>selector.includes('#panel-nav-primary')?host?.nav:null,
  addEventListener:(name,fn)=>add(docEvents,name,fn),
  createElement(tag){
   if(tag==='template')return {set innerHTML(markup){this.nav=navFrom(markup)},get content(){return{querySelector:()=>this.nav}}};
   const e=element();e.fallback=element();e.querySelector=selector=>selector==='.standalone-nav-fallback'?e.fallback:null;e.replaceChildren=nav=>{e.nav=nav;e.fallback=null};return e;
  }};
 function setBody(){document.body=element();document.body.prepend=e=>host=e}
 if(bodyReady)setBody();
 const window={location,innerWidth:1200,innerHeight:800,addEventListener:(n,f)=>add(windowEvents,n,f),
  setTimeout:(fn,delay)=>{const id=++nextTimer;timers.set(id,{fn,delay});return id},clearTimeout:id=>timers.delete(id)};
 const c=vm.createContext({window,document,AbortController,URL,console:{warn:(...a)=>warnings.push(a)},
  sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
  fetch:async url=>{requests.push(url);const response=responses[Math.min(requests.length-1,responses.length-1)];return{ok:response===200,status:response,text:async()=>menu}}});
 vm.runInContext(source,c);
 return {document,window,requests,warnings,routes,storage,timers,get host(){return host},
  install(){setBody();emit(docEvents,'DOMContentLoaded')},emit:(name,event)=>emit(windowEvents,name,event),
  async retry(){await settle();const pair=[...timers].find(([,t])=>t.delay<15000);assert(pair,'pending retry timer');timers.delete(pair[0]);pair[1].fn();await settle()}};
}
test('navigation starts early, recovers transient failure, preserves all real menu routes',async()=>{
 const h=harness({bodyReady:false,responses:[503,200]});assert.deepEqual(h.requests,['/ui/module-menu']);assert.equal(h.host,undefined);
 h.install();await h.retry();assert.equal(h.requests.length,2);assert.equal(h.host.nav.links.length,35);
 const expected=[...menu.matchAll(/<a\b[^>]*href="([^"]*)"/g)].map(m=>m[1]);
 assert.deepEqual(h.host.nav.links.map(link=>link.href),expected);assert(h.host.nav.links.every(link=>link.target==='_self'));
 assert(expected.some(url=>url.includes('?')),'retain query parameters in module routes');assert(expected.some(url=>url.startsWith('/#')),'retain in-page destinations');
 const click=h.host.nav.handlers.click[0],trigger={dataset:{directTarget:'/rl-panel?from=hub&confirm=prompt'},matches:()=>true};
 click({target:{closest:selector=>selector==='.nav-trigger'?trigger:null}});assert.deepEqual(h.routes,['/rl-panel?from=hub&confirm=prompt']);
});
test('cached menu is immediate and survives three failed refreshes with current route marked',async()=>{
 const h=harness({cache:menu,path:'/ops-copilot?mission=handoff',responses:[503]});assert.equal(h.host.nav.links.length,35);
 assert.equal(h.host.nav.querySelector('[aria-current="page"]').attrs.href,'/ops-copilot');
 await h.retry();await h.retry();assert.equal(h.requests.length,3);assert.equal(h.host.nav.links.length,35);assert.equal(h.host.fallback,null);assert.equal(h.timers.size,0);
});
test('uncached failed navigation reaches explicit bounded fallback',async()=>{
 const h=harness({responses:[503]});await h.retry();await h.retry();assert.equal(h.requests.length,3);
 assert.match(h.host.fallback.textContent,/菜单暂未加载/);assert.equal(h.timers.size,0);assert.equal(h.host.nav,undefined);
});
test('pagehide cancels a retry and persisted pageshow restores an absent menu',async()=>{
 const responses=[503,200],h=harness({responses});await settle();h.emit('pagehide');await settle();assert.equal(h.requests.length,1);assert.equal(h.timers.size,0);
 h.emit('pageshow',{persisted:true});await settle();assert.equal(h.requests.length,2);assert.equal(h.host.nav.links.length,35);
});
test('malformed cached URL never prevents recovery from fresh valid markup',async()=>{
 const h=harness({cache:'<nav id="panel-nav-primary"><a href="http://[invalid">broken</a></nav>'});await settle();
 assert.equal(h.host.nav.links.length,35);assert.equal(h.storage.get(cacheKey),menu);assert.equal(h.warnings.length,1);
});
