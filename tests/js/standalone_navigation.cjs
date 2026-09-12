// Run with Node.js + Playwright; PLAYWRIGHT_CHANNEL=chrome may use an installed Chrome.
// Network responses are controlled so failure/retry/lifecycle checks are reproducible.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');
const base = require('node:path').resolve(__dirname, '../..');
const source = fs.readFileSync(base + '/app/ui/index.html', 'utf8');
const menu = source.match(/<nav\b(?=[^>]*\bid=["']panel-nav-primary["'])[^>]*>[\s\S]*?<\/nav>/)[0].replace(/\b(href|data-direct-target)=(["'])#/g, '$1=$2/#');
const script = fs.readFileSync(base + '/app/static/standalone_navigation.js', 'utf8');
const html = '<!doctype html><html><head><meta charset="utf-8"><script src="/static/standalone_navigation.js"></script></head><body><main>Module body</main><script>fetch("/business-marker")</script></body></html>';
(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    let count = 0;
    let failAll = false;
    const requests = [];
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const path = new URL(route.request().url()).pathname;
      requests.push(path);
      if (path === '/static/standalone_navigation.js') return route.fulfill({ contentType: 'text/javascript', body: script });
      if (path === '/ui/module-menu') {
        count += 1;
        return route.fulfill({ status: failAll || count < 2 ? 503 : 200, contentType: 'text/html', body: menu });
      }
      if (path === '/business-marker') return route.fulfill({ contentType: 'application/json', body: '{}' });
      return route.fulfill({ contentType: 'text/html', body: html });
    });
    await page.goto('http://standalone-menu.test/v3');
    assert(requests.indexOf('/ui/module-menu') < requests.indexOf('/business-marker'));
    await page.waitForSelector('#standalone-module-navigation #panel-nav-primary');
    assert.equal(count, 2, 'first entry should recover after the transient error');
    assert.equal(await page.locator('#standalone-module-navigation a').count(), 35);
    failAll = true;
    count = 0;
    await page.goto('http://standalone-menu.test/ops-copilot');
    assert.equal(await page.locator('#standalone-module-navigation a').count(), 35, 'cached menu must already be present at DOM ready');
    assert.equal(await page.locator('[aria-current="page"]').getAttribute('href'), '/ops-copilot');
    await page.waitForTimeout(3600);
    assert.equal(count, 3, 'refresh retry count must be bounded');
    assert.equal(await page.locator('#standalone-module-navigation a').count(), 35, 'failed refresh must retain cached menu');
    assert.equal(await page.locator('.standalone-nav-fallback').count(), 0);
    await page.evaluate(() => sessionStorage.clear());
    count = 0;
    await page.goto('http://standalone-menu.test/rl-panel');
    await page.waitForFunction(() => document.querySelector('.standalone-nav-fallback')?.textContent.includes('菜单暂未加载'));
    assert.equal(count, 3);
    count = 0;
    await page.goto('http://standalone-menu.test/rl-panel');
    await page.waitForFunction(() => document.readyState === 'complete');
    await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true })));
    await page.waitForTimeout(1200);
    const requestsAtHide = count;
    assert.equal(requestsAtHide, 1, 'pagehide must stop a pending retry');
    failAll = false;
    await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true })));
    await page.waitForSelector('#standalone-module-navigation #panel-nav-primary');
    assert.equal(count, 2, 'a BFcache-style restoration without a menu must retry');
    await page.evaluate(() => sessionStorage.setItem('port-dt.module-menu.v20260912-2', '<nav id="panel-nav-primary"><a href="http://[invalid">broken cache</a></nav>'));
    count = 0;
    await page.goto('http://standalone-menu.test/integration-hub');
    await page.waitForSelector('#standalone-module-navigation #panel-nav-primary');
    assert.equal(await page.locator('#standalone-module-navigation a').count(), 35, 'invalid cached markup must recover from the fresh menu');
    assert.deepEqual(errors, [], 'failed menu requests must not raise page errors or unhandled promises');
    console.log('PASS: early head request, retry recovery, immediate session cache, retained cache on refresh failure, bounded retry/fallback, pagehide cancellation and persisted pageshow recovery, no unhandled errors.');
    await context.close();
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
