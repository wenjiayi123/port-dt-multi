(function () {
  'use strict';
  const homeId = 'home-hero';
  const links = Array.from(document.querySelectorAll('#panel-nav-primary .nav-dropdown a[href^="#"]'));
  const pages = new Map([[homeId, {node: document.getElementById(homeId), title: '首页总览'}]]);
  links.forEach(link => {
    const id = link.dataset.targetId || link.getAttribute('href').slice(1);
    const node = document.getElementById(id);
    if (node) pages.set(id, {node, link, title: link.firstElementChild.textContent.trim()});
  });
  const hidden = new Set();
  const shells = new Set();
  let active = null;
  let revision = 0;
  let ready = false;
  const title = document.getElementById('module-page-title');
  const baseTitle = document.title;

  function resolve(value) {
    let id;
    try { id = decodeURIComponent(String(value || homeId).replace(/^#/, '')); }
    catch (_) { return null; }
    let node = document.getElementById(id);
    while (node && node !== document.body) {
      const page = pages.get(node.id);
      if (page) return node.id === homeId || page.link.hasAttribute('data-module-page') ? page : null;
      node = node.parentElement;
    }
    return null;
  }

  function conceal(node) {
    // Router classes never change a tab's own hidden/style/disabled state.
    if (['SCRIPT', 'STYLE', 'LINK', 'TEMPLATE'].includes(node.tagName)) return;
    node.classList.add('module-route-hidden');
    hidden.add(node);
  }

  function open(value, options = {}) {
    if (!ready) return false;
    const page = resolve(value);
    if (!page) return false;
    const node = page.node;
    const previous = active;
    const changed = active !== node;
    hidden.forEach(el => el.classList.remove('module-route-hidden'));
    shells.forEach(el => el.classList.remove('module-route-shell'));
    hidden.clear();
    shells.clear();
    active?.classList.remove('module-route-active');

    pages.forEach(other => {
      if (other.node !== node && !other.node.contains(node)) conceal(other.node);
    });
    // A module may be nested several layers inside another module. Keep only
    // its ancestor path visible; do not move, clone or recreate business DOM.
    for (let child = node, parent = node.parentElement; parent && parent !== document.body; child = parent, parent = parent.parentElement) {
      parent.classList.add('module-route-shell');
      shells.add(parent);
      Array.from(parent.children).forEach(sibling => { if (sibling !== child) conceal(sibling); });
    }
    active = node;
    node.classList.add('module-route-active');
    document.body.dataset.moduleView = node.id;
    title.textContent = page.title;
    document.title = node.id === homeId ? baseTitle : `${page.title} · 港口 AI 运营平台`;
    document.querySelectorAll('.nav-group.is-open, .nav-trigger.is-open').forEach(el => el.classList.remove('is-open'));
    document.querySelectorAll('.nav-trigger[aria-expanded]').forEach(el => el.setAttribute('aria-expanded', 'false'));
    links.forEach(link => {
      if (link === page.link) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    document.querySelectorAll('#panel-nav-primary .nav-trigger').forEach(trigger => {
      const current = page.link ? trigger === page.link.closest('.nav-group').querySelector('.nav-trigger') : trigger.dataset.directTarget === '#home-hero';
      trigger.classList.toggle('is-current', current);
    });
    if (changed && previous) {
      window.dispatchEvent(new CustomEvent('port-module-hidden', {detail: {id: previous.id, nextId: node.id}}));
    }
    const mode = options.history || 'push';
    const hash = `#${node.id}`;
    if (mode !== 'none' && location.hash !== hash) {
      history[mode === 'replace' ? 'replaceState' : 'pushState'](history.state, '', hash);
    }
    if (changed || options.focus !== false) window.scrollTo({top: 0, behavior: 'instant'});
    if (options.focus !== false) {
      node.setAttribute('tabindex', '-1');
      node.focus({preventScroll: true});
    }
    const token = ++revision;
    requestAnimationFrame(() => {
      if (token !== revision) return;
      window.dispatchEvent(new Event('resize'));
      node.querySelectorAll('[_echarts_instance_]').forEach(el => {
        if (el.getBoundingClientRect().width > 0) window.echarts?.getInstanceByDom(el)?.resize();
      });
      window.dispatchEvent(new CustomEvent('port-module-shown', {detail: {id: node.id}}));
    });
    return true;
  }

  window.PortModuleNavigation = Object.freeze({open, get current() { return active?.id || homeId; }});

  function init() {
    ready = true;
    // Seed the shared menu before the first standalone navigation, so slow
    // evidence APIs cannot delay the user's ability to switch modules again.
    try {
      const menu = document.getElementById('panel-nav-primary').cloneNode(true);
      menu.querySelectorAll('a[href^="#"]').forEach(link => link.setAttribute('href', '/' + link.getAttribute('href')));
      menu.querySelectorAll('[data-direct-target^="#"]').forEach(button => { button.dataset.directTarget = '/' + button.dataset.directTarget; });
      sessionStorage.setItem('port-dt.module-menu.v20260912-2', menu.outerHTML);
    } catch (_) { /* Private browser storage may be unavailable. */ }
    const initial = location.hash || homeId;
    if (!open(initial, {history: 'replace', focus: false})) open(homeId, {history: 'replace', focus: false});
    document.addEventListener('click', event => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const link = event.target.closest('a[href^="#"]');
      if (!link || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
      if (open(link.getAttribute('href'))) event.preventDefault();
    });
    const restore = () => {
      if (!open(location.hash || homeId, {history: 'none'})) open(homeId, {history: 'replace'});
    };
    window.addEventListener('popstate', restore);
    window.addEventListener('hashchange', restore);
  }
  if (document.readyState !== 'complete') document.addEventListener('DOMContentLoaded', init, {once: true});
  else init();
})();
