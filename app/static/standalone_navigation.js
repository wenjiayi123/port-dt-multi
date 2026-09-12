/* Shared, same-tab navigation for the six independent module documents. */
(function () {
  'use strict';
  if (window.__standaloneNavigationRequested) return;
  window.__standaloneNavigationRequested = true;

  // Start in <head>, before the business scripts can queue slower API calls.
  // Settle failures into a value immediately: DOMContentLoaded may be delayed.
  const cacheKey = 'port-dt.module-menu.v20260912-2';
  let cachedHTML = '';
  try { cachedHTML = sessionStorage.getItem(cacheKey) || ''; } catch (_) {}
  let requestController = null;
  let retryTimer = null;
  let resolveRetry = null;
  let leaving = false;
  let generation = 0;
  let refreshMenu = null;
  async function requestMenu() {
    let lastError;
    const requestGeneration = generation;
    for (let attempt = 0; attempt < 3 && !leaving && requestGeneration === generation; attempt += 1) {
      requestController = new AbortController();
      const controller = requestController;
      const timeout = window.setTimeout(function () { controller.abort(); }, 15000);
      try {
        const response = await fetch('/ui/module-menu', { signal: controller.signal, cache: 'no-cache' });
        if (!response.ok) throw new Error('Menu unavailable (HTTP ' + response.status + ')');
        const html = await response.text();
        if (leaving || requestGeneration !== generation) return { cancelled: true };
        return { html: html };
      } catch (error) {
        lastError = error;
      } finally {
        window.clearTimeout(timeout);
      }
      if (attempt < 2 && !leaving && requestGeneration === generation) {
        await new Promise(function (resolve) {
          resolveRetry = resolve;
          retryTimer = window.setTimeout(function () {
            resolveRetry = null;
            resolve();
          }, 1000 * (attempt + 1));
        });
      }
    }
    if (leaving || requestGeneration !== generation) return { cancelled: true };
    return { error: lastError };
  }
  const menuRequest = requestMenu().catch(function (error) { return { error: error }; });
  window.addEventListener('pagehide', function () {
    leaving = true;
    generation += 1;
    if (requestController) requestController.abort();
    window.clearTimeout(retryTimer);
    if (resolveRetry) {
      resolveRetry();
      resolveRetry = null;
    }
  });
  window.addEventListener('pageshow', function (event) {
    if (!event.persisted) return;
    leaving = false;
    if (refreshMenu && !document.querySelector('#standalone-module-navigation #panel-nav-primary')) refreshMenu();
  });

  function install() {
    if (document.getElementById('standalone-module-navigation')) return;

    const host = document.createElement('div');
    host.id = 'standalone-module-navigation';
    host.setAttribute('aria-label', '模块导航');
    host.innerHTML = '<a class="standalone-nav-fallback" href="/">← 返回主界面</a>';
    document.body.prepend(host);
    document.body.classList.add('has-standalone-navigation');
    let opened = null;
    let renderedHTML = '';

    function measure() {
      document.documentElement.style.setProperty('--standalone-navigation-height', host.offsetHeight + 'px');
      if (opened) positionDropdown(opened);
    }

    function positionDropdown(group) {
      const dropdown = group.querySelector('.nav-dropdown');
      const trigger = group.querySelector('.nav-trigger');
      if (!dropdown || !trigger) return;
      const rect = trigger.getBoundingClientRect();
      const top = host.getBoundingClientRect().bottom + 8;
      const width = dropdown.offsetWidth;
      dropdown.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)) + 'px';
      dropdown.style.top = top + 'px';
      dropdown.style.maxHeight = Math.max(80, window.innerHeight - top - 12) + 'px';
    }

    function closeDropdown(restoreFocus) {
      if (!opened) return;
      const trigger = opened.querySelector('.nav-trigger');
      opened.querySelector('.nav-dropdown').hidden = true;
      opened.classList.remove('is-open');
      trigger.setAttribute('aria-expanded', 'false');
      opened = null;
      if (restoreFocus) trigger.focus();
    }

    function openDropdown(group, focusLast) {
      closeDropdown(false);
      const dropdown = group.querySelector('.nav-dropdown');
      if (!dropdown) return;
      opened = group;
      dropdown.hidden = false;
      group.classList.add('is-open');
      group.querySelector('.nav-trigger').setAttribute('aria-expanded', 'true');
      positionDropdown(group);
      if (focusLast !== undefined) {
        const links = dropdown.querySelectorAll('a');
        const target = focusLast ? links[links.length - 1] : links[0];
        if (target) target.focus();
      }
    }

    function wireMenu(nav) {
      nav.querySelectorAll('.nav-group').forEach(function (group, index) {
        const trigger = group.querySelector('.nav-trigger');
        const dropdown = group.querySelector('.nav-dropdown');
        group.classList.remove('is-open');
        if (trigger) trigger.classList.remove('is-open', 'is-current');
        if (trigger && dropdown) {
          dropdown.id = 'standalone-nav-group-' + index;
          dropdown.hidden = true;
          trigger.setAttribute('aria-expanded', 'false');
          trigger.setAttribute('aria-controls', dropdown.id);
        }
      });
      nav.querySelectorAll('a[href]').forEach(function (link) {
        // Regular links replace this document: its timers, streams, and visuals
        // cannot remain active behind the next module as they would in an iframe.
        link.target = '_self';
        link.removeAttribute('aria-current');
        const target = new URL(link.href, window.location.href);
        if (target.pathname.replace(/\/$/, '') === window.location.pathname.replace(/\/$/, '')) {
          link.setAttribute('aria-current', 'page');
        }
      });
      nav.addEventListener('click', function (event) {
        const trigger = event.target.closest('.nav-trigger');
        if (trigger && trigger.matches('button[data-direct-target]')) {
          window.location.assign(trigger.dataset.directTarget);
          return;
        }
        const group = trigger && trigger.closest('.nav-group');
        if (group && group.querySelector('.nav-dropdown')) {
          event.preventDefault();
          if (opened === group) closeDropdown(false);
          else openDropdown(group);
        } else if (event.target.closest('a')) {
          closeDropdown(false);
        }
      });
      nav.addEventListener('keydown', function (event) {
        const trigger = event.target.closest('.nav-trigger');
        if (trigger && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
          const group = trigger.closest('.nav-group');
          if (!group.querySelector('.nav-dropdown')) return;
          event.preventDefault();
          openDropdown(group, event.key === 'ArrowUp');
        }
      });
      nav.addEventListener('scroll', function () { closeDropdown(false); }, { passive: true });
    }

    document.addEventListener('pointerdown', function (event) {
      if (!host.contains(event.target)) closeDropdown(false);
    });
    document.addEventListener('focusin', function (event) {
      if (opened && !opened.contains(event.target)) closeDropdown(false);
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && opened) {
        event.preventDefault();
        closeDropdown(true);
      }
    });
    window.addEventListener('resize', measure, { passive: true });
    if (typeof ResizeObserver === 'function') new ResizeObserver(measure).observe(host);
    window.addEventListener('pagehide', function () { closeDropdown(false); });
    measure();

    function renderMenu(html) {
      if (!html || html.length > 64000) return false;
      if (html === renderedHTML) return true;
      const template = document.createElement('template');
      template.innerHTML = html;
      const nav = template.content.querySelector('#panel-nav-primary');
      if (!nav || !nav.querySelector('a[href]')) return false;
      closeDropdown(false);
      wireMenu(nav);
      host.replaceChildren(nav);
      renderedHTML = html;
      measure();
      // The node must be attached before its scroll geometry is available.
      const active = nav.querySelector('[aria-current="page"]');
      if (active && nav.scrollWidth > nav.clientWidth) {
        nav.scrollLeft = Math.max(0, active.offsetLeft - (nav.clientWidth - active.offsetWidth) / 2);
      }
      return true;
    }

    function showUnavailable(error) {
      // A failed refresh must never remove an already usable cached menu.
      const fallback = host.querySelector('.standalone-nav-fallback');
      if (fallback) fallback.textContent = '← 返回主界面 · 菜单暂未加载';
      if (error) console.warn('Module navigation unavailable:', error);
    }

    try { if (cachedHTML) renderMenu(cachedHTML); } catch (error) { showUnavailable(error); }
    function applyResult(result) {
      if (result.cancelled || leaving) return;
      if (result.html && renderMenu(result.html)) {
        try { sessionStorage.setItem(cacheKey, result.html); } catch (_) {}
      } else {
        showUnavailable(result.error);
      }
    }
    refreshMenu = function () { requestMenu().then(applyResult).catch(showUnavailable); };
    menuRequest.then(applyResult).catch(showUnavailable);
  }

  if (document.body) install();
  else document.addEventListener('DOMContentLoaded', install, { once: true });
})();
