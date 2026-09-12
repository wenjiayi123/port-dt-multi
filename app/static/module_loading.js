/* Page-owned loading. Unopened modules remain idle; cached UI is never an admission gate. */
(function () {
  'use strict';
  if (window.PortModuleLoading) return;
  const entries = new Map();
  let ready = false;
  let nextKey = 0;
  let scheduled = false;
  let resolveDOM;
  const routedDOMReady=new Promise(resolve=>{resolveDOM=resolve;});
  const now = () => Date.now();
  function current() {
    return window.PortModuleNavigation?.current || document.body?.dataset.moduleView || 'home-hero';
  }
  function isActive(ids) {
    const active = current();
    return ids.includes(active);
  }
  function notify(entry) {
    window.dispatchEvent(new CustomEvent('port-module-load-state', {detail:{key:entry.key, sections:entry.sections.slice(), state:entry.state, completedAt:entry.completedAt}}));
  }
  function run(entry, force=false) {
    if (entry.pending) return entry.pending;
    if (!force && entry.state==='loaded' && now()-entry.completedAt < entry.maxAgeMs) return Promise.resolve(entry.value);
    entry.state='loading';notify(entry);
    entry.pending=Promise.resolve().then(entry.task).then(value=>{
      entry.state='loaded';entry.completedAt=now();entry.value=value;notify(entry);return value;
    },error=>{entry.state='failed';notify(entry);throw error;}).finally(()=>{entry.pending=null;});
    return entry.pending;
  }
  function sync() {
    scheduled=false;
    if (!ready || document.hidden) return;
    entries.forEach(entry=>{
      if(isActive(entry.sections)) run(entry).catch(error=>console.warn('[module-load]',entry.key,error));
    });
  }
  function schedule() {
    if(scheduled) return;
    scheduled=true;
    // The router's DOMContentLoaded handler must finish concealing other pages first.
    window.setTimeout(sync,0);
  }
  function register(sections, task, options={}) {
    const ids=Array.isArray(sections)?sections.slice():[sections];
    if(!ids.length || ids.some(id=>typeof id!=='string') || typeof task!=='function') throw new TypeError('Module loading requires section ids and a task');
    const key=options.key || `${ids.join(',')}:${++nextKey}`;
    if(entries.has(key)) throw new Error(`Duplicate module loader: ${key}`);
    const entry={key,sections:ids,task,state:'idle',pending:null,completedAt:null,value:undefined,maxAgeMs:options.maxAgeMs??60000};
    entries.set(key,entry);
    if(ready)schedule();
    return ()=>run(entry,true);
  }
  window.PortModuleLoading=Object.freeze({register,current,isActive:section=>isActive(Array.isArray(section)?section:[section]),
    snapshot:()=>Array.from(entries.values(),entry=>({key:entry.key,sections:entry.sections.slice(),state:entry.state,completedAt:entry.completedAt})),
    refresh:key=>{const entry=entries.get(key);return entry?run(entry,true):Promise.reject(new Error('Unknown module loader'));}});
  window.__portDtLoadWhenActive=register;
  window.addEventListener('port-module-shown',schedule);
  window.addEventListener('pageshow',schedule);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)schedule();});
  const start=()=>{ready=true;schedule();window.setTimeout(resolveDOM,0);};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();

  // Share only simultaneous, identical, same-origin API GETs. Never persist a
  // response or merge writes, credentials, abort scopes, or different queries.
  const nativeFetch=window.fetch.bind(window);
  const pendingGets=new Map();
  window.fetch=function(input, init={}) {
    if((typeof input!=='string' && !(input instanceof URL)) || init.signal || (init.method||'GET').toUpperCase()!=='GET') return nativeFetch(input,init);
    let url;
    try{url=new URL(String(input),window.location.href);}catch(_){return nativeFetch(input,init);}
    if(url.origin!==window.location.origin || !url.pathname.startsWith('/api/')) return nativeFetch(input,init);
    const headers=Array.from(new Headers(init.headers||{}).entries()).sort((a,b)=>a[0].localeCompare(b[0]));
    const key=JSON.stringify([url.href,headers,init.credentials||'same-origin',init.cache||'default',init.mode||'cors',init.redirect||'follow',init.referrer,init.referrerPolicy,init.integrity]);
    let pending=pendingGets.get(key);
    if(!pending){
      pending=Promise.resolve().then(async()=>{
        // Do not occupy the browser's API connections before deferred router
        // and recovery scripts have finished parsing. The lightweight source
        // summary is the sole early API exception; full hash checks stay gated.
        if(!ready && !(url.pathname==='/api/system/provenance' && url.searchParams.get('detail')==='summary'))await routedDOMReady;
        return nativeFetch(input,init);
      });
      pendingGets.set(key,pending);
      pending.then(()=>{if(pendingGets.get(key)===pending)pendingGets.delete(key);},()=>{if(pendingGets.get(key)===pending)pendingGets.delete(key);});
    }
    return pending.then(response=>response.clone());
  };
})();
