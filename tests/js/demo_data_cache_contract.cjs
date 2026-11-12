// Actual helper code with controlled HTTP promises; no claim of browser coverage.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
function setup(){
 const requests=[],events={};
 const document={readyState:'loading',addEventListener(name,fn){events[name]=fn;}};
 const window={addEventListener(name,fn){events[name]=fn;}};
 const context={window,document,Date,Number,String,Promise,AbortController,setTimeout,clearTimeout,fetch:url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject}))};
 vm.runInNewContext(fs.readFileSync('app/static/demo-data-workspace.js','utf8'),context);
 return{load:window.PortDemoData.load,requests};
}
const valid={schema:'port-dt-demo-workspace.v1',snapshot:{profile:{timezone:'Asia/Shanghai'}},window:{},annual:{},coverage:{},metrics:[],site_pending:[]};
test('no API request at helper startup; shared pending fetch and one-minute cache avoid duplicate work',async()=>{
 const h=setup();assert.equal(h.requests.length,0);
 const a=h.load(),b=h.load();assert.equal(a,b);assert.equal(h.requests.length,1);
 h.requests[0].resolve({ok:true,json:async()=>valid});assert.deepEqual(await a,valid);await b;
 assert.deepEqual(await h.load(),valid);assert.equal(h.requests.length,1);
});
test('failed explicit refresh discards cached success and permits a real retry',async()=>{
 const h=setup();const a=h.load();h.requests[0].resolve({ok:true,json:async()=>valid});await a;
 const failed=h.load(true);h.requests[1].resolve({ok:false,status:503});await assert.rejects(failed,/503/);
 const retry=h.load();assert.equal(h.requests.length,3);h.requests[2].resolve({ok:true,json:async()=>valid});await retry;
});
test('malformed business response is never cached as a successful source',async()=>{
 const h=setup();const a=h.load();h.requests[0].resolve({ok:true,json:async()=>({schema:'port-dt-demo-workspace.v1',metrics:[]})});await assert.rejects(a,/契约/);
 const retry=h.load();assert.equal(h.requests.length,2);h.requests[1].resolve({ok:true,json:async()=>valid});await retry;
});
