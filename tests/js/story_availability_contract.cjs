// Actual story controller with offline DOM/HTTP fixtures, no browser.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.resolve(__dirname,'../../app/ui/index.html'),'utf8');
const script=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].find(m=>m[1].includes('let __storyRequestRun = 0;'))[1];
const nodes=new Map(),events={},timers=new Set();let nextTimer=0;
const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',innerHTML:'',disabled:false,dataset:{},handlers:{},addEventListener(type,fn){this.handlers[type]=fn;},getClientRects(){return[{}];},appendChild(){}});return nodes.get(id);};
node('story-port').value='shanghai';node('story-scenario').value='sac_vs_fcfs';
let response={available:true,hour:0,events:[],baseline:{},policy:{}};
const context={JSON,Number,String,Math,Promise,console,document:{hidden:false,getElementById:node,createElement:()=>node('created'),addEventListener(){}},window:{addEventListener(type,fn){events[type]=fn;}},fetch:async()=>({ok:true,json:async()=>response}),setInterval(){const id=++nextTimer;timers.add(id);return id;},clearInterval(id){timers.delete(id);}};
vm.createContext(context);vm.runInContext(script,context);
const settle=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 events.DOMContentLoaded();await settle();assert.equal(node('story-play').disabled,false);assert.equal(node('story-hour-label').textContent,'T');
 await node('story-play').handlers.click();await settle();assert.ok(timers.size>0);
 response={available:false,reason:'no verified trajectory'};node('story-port').value='ningbo';node('story-port').handlers.change();await settle();assert.equal(node('story-hour-label').textContent,'—');assert.equal(node('story-play').disabled,true);assert.equal(timers.size,0);assert.match(node('story-play').textContent,/待接入/);
 await node('story-play').handlers.click();assert.equal(timers.size,0,'programmatic click cannot start unavailable story');
 response={available:true,hour:3,events:[],baseline:{},policy:{}};node('story-port').value='shanghai';node('story-port').handlers.change();await settle();assert.equal(node('story-hour-label').textContent,'T+3h');assert.equal(node('story-play').disabled,false);
 await node('story-play').handlers.click();await settle();assert.ok(timers.size>0);
 response={available:true,hour:null};await context.window.loadStory(0);assert.equal(node('story-hour-label').textContent,'—');assert.equal(node('story-play').disabled,true);assert.equal(timers.size,0);
 console.log('STORY_AVAILABILITY_CONTRACT:PASS:4 (offline DOM/HTTP fixtures; no browser claim)');
})().catch(error=>{console.error(error);process.exitCode=1;});
