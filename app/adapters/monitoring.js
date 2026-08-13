/* ============================================================================
 * app/adapters/monitoring.js
 * ----------------------------------------------------------------------------
 * 【大白话】前端“监测与运维”适配器：
 * 将后端接口 /api/monitoring/* 封装成简单方法，统一处理时间/编码/错误。
 *
 * 暴露到 window.MonitoringAPI：
 *   - anomalyScan({ assetIds, start, end, windowMin, stepSec, method, sensitivity,
 *                   residual, point, assetType })
 *   - psi({ assetId, point, baselineStart, baselineEnd, recentStart, recentEnd,
 *           bins, stepSec })
 *
 * 真实港口落地：后端只需保证 /api/monitoring/* 与 di.telemetry.get_series(...) 接好；
 *               此适配器无需改动。
 * ========================================================================== */

(function attachMonitoringAdapter(global){
  'use strict';

  // ---------- 基础工具 ----------
  const toISO = (x) => {
    if (x == null || x === '') return '';
    if (typeof x === 'number') {
      // epoch 秒
      return new Date(x * 1000).toISOString();
    }
    // 字符串：如果是纯数字，按 epoch 秒；否则认为是 ISO
    const s = String(x).trim();
    if (/^\d+(\.\d+)?$/.test(s)) {
      return new Date(Number(s) * 1000).toISOString();
    }
    // 宽松校验：让后端自己 parse ISO/epoch
    return s;
  };

  const nowISO = () => new Date().toISOString();
  const minusMinISO = (m) => new Date(Date.now() - m * 60000).toISOString();

  const enc = (o) => new URLSearchParams(o).toString();

  async function getJSON(url){
    const r = await fetch(url);
    if (!r.ok) {
      const t = await r.text().catch(()=> '');
      throw new Error(`HTTP ${r.status}: ${url}\n${t.slice(0, 200)}`);
    }
    return await r.json();
  }

  // ---------- 适配器：异常扫描 ----------
  /**
   * anomalyScan
   * @param {Object} opt
   *   - assetIds: string | string[]  （资产ID；单个或数组；默认 ['qc-01']）
   *   - start, end: string|number    （ISO 或 epoch 秒；留空=最近30分钟）
   *   - windowMin: number            （可选；若未传 start/end 则使用它，默认 30）
   *   - stepSec: number              （默认 60）
   *   - method: 'iqr'|'zscore'|'ewma'（默认 'iqr'）
   *   - sensitivity: number          （默认 1.5；iqr=k, zscore=z, ewma=k）
   *   - residual: boolean            （默认 false；如需残差异常，可设 true）
   *   - point: string                （默认 'active_power_kw'）
   *   - assetType: string            （默认 'quay_crane'）
   * @returns Promise<object> 后端 JSON
   */
  async function anomalyScan(opt = {}){
    const {
      assetIds = ['qc-01'],
      start = '',
      end = '',
      windowMin = 30,
      stepSec = 60,
      method = 'iqr',
      sensitivity = 1.5,
      residual = false,
      point = 'active_power_kw',
      assetType = 'quay_crane',
    } = opt;

    // 资产参数：支持单个字符串或数组
    const assets = Array.isArray(assetIds) ? assetIds.join(',') : String(assetIds||'qc-01');

    // 时间窗口：优先使用显式 start/end；否则用 windowMin
    let s = (start && toISO(start)) || '';
    let e = (end && toISO(end)) || '';
    if (!s || !e) {
      s = minusMinISO(windowMin || 30);
      e = nowISO();
    }

    // 组装查询参数
    // 说明：API 已支持 ISO/epoch；这里传 ISO，落地时便于跨时区对齐
    const q = enc({
      assets, start: s, end: e,
      method, sensitivity, step_sec: stepSec,
      point, asset_type: assetType,
      residual: residual ? 1 : 0,
    });

    const url = `/api/monitoring/anomaly/scan?${q}`;
    return await getJSON(url);
  }

  // ---------- 适配器：PSI 漂移 ----------
  /**
   * psi
   * @param {Object} opt
   *   - assetId: string              （必填）
   *   - point: string                （默认 'active_power_kw'）
   *   - baselineStart, baselineEnd   （ISO/epoch；留空=结束前60~30分钟）
   *   - recentStart, recentEnd       （ISO/epoch；留空=结束前30分钟~结束）
   *   - bins: number                 （默认 10）
   *   - stepSec: number              （默认 60）
   * @returns Promise<object> 后端 JSON
   */
  async function psi(opt = {}){
    const {
      assetId = 'qc-01',
      point = 'active_power_kw',
      baselineStart = '',
      baselineEnd = '',
      recentStart = '',
      recentEnd = '',
      bins = 10,
      stepSec = 60,
    } = opt;

    // 结束时间：优先用 recentEnd，否则取当前
    const endISO = recentEnd ? toISO(recentEnd) : nowISO();
    const b0 = baselineStart ? toISO(baselineStart) : minusMinISO(60);
    const b1 = baselineEnd   ? toISO(baselineEnd)   : minusMinISO(30);
    const r0 = recentStart   ? toISO(recentStart)   : minusMinISO(30);
    const r1 = endISO;

    const q = enc({
      asset_id: assetId,
      point,
      baseline_start: b0,
      baseline_end: b1,
      recent_start: r0,
      recent_end: r1,
      bins,
      step_sec: stepSec,
    });

    const url = `/api/monitoring/drift/psi?${q}`;
    return await getJSON(url);
  }

  // ---------- 可选：最小化的 DOM 绑定（可不使用） ----------
  /**
   * 把按钮与页面元素绑定在一起（如果你用我的 index.html 新增区块的ID）
   * 使用方式：
   *   MonitoringAPI.bindDefaultUI();
   */
  function bindDefaultUI(){
    const $ = (id) => document.getElementById(id);
    const fmt = (x, n=3) => (Number.isFinite(Number(x)) ? Number(x).toFixed(n) : '—');

    const btnAnom = $('btn-anom');
    const btnPSI  = $('btn-psi');

    async function renderAnom(){
      try{
        const asset = ($('mon-asset')?.value || 'qc-01').trim();
        const method = ($('mon-method')?.value || 'iqr').trim();
        const sens = Number($('mon-sens')?.value || 1.5);
        const start = $('mon-start')?.value || '';
        const end   = $('mon-end')?.value || '';
        const res = await anomalyScan({ assetIds: asset, method, sensitivity: sens, start, end, stepSec: 60 });
        const items = Array.isArray(res.items)? res.items : [];
        const total = items.reduce((a,b)=> a + ((b.anomalies||[]).length), 0);
        const sum = $('mon-anom-summary');
        if (sum) sum.textContent = `资产数=${items.length}，异常总数=${total}` + (res.audit_uri?`（证据：${res.audit_uri}）`:'');
        const tb = $('mon-anom-table')?.querySelector('tbody');
        if (tb){
          const rows = items.flatMap(it => (it.anomalies||[]).slice(0,200).map(a=>(
            `<tr>
              <td>${it.asset_id||it.asset||asset}</td>
              <td class="mono">${a.ts}</td>
              <td class="mono">${fmt(a.v,3)}</td>
              <td>${fmt(a.score,2)}</td>
              <td>${a.reason||'—'}</td>
            </tr>`
          )));
          tb.innerHTML = rows.length? rows.join('') : `<tr><td colspan="5" class="small">暂无异常</td></tr>`;
        }
      }catch(e){
        const sum = $('mon-anom-summary');
        if (sum) sum.textContent = '请求失败：' + e.message;
      }
    }

    async function renderPSI(){
      try{
        const asset = ($('mon-asset')?.value || 'qc-01').trim();
        const res = await psi({ assetId: asset, point: 'active_power_kw', bins: 10, stepSec: 60 });
        const sum = $('mon-psi-summary');
        if (sum) sum.textContent = `PSI=${fmt(res.psi,3)} 级别=${res.level||'—'}（基线样本=${res.baseline?.n||0}，近期样本=${res.recent?.n||0}）`;
        // 简易直方对比
        const cvs = $('mon-psi-canvas'); const ctx = cvs?.getContext('2d');
        if (ctx){
          ctx.clearRect(0,0,cvs.width,cvs.height);
          const bins = Array.isArray(res.bins)? res.bins : [];
          if (bins.length){
            const H=cvs.height, W=cvs.width, pad=30;
            const maxP = Math.max(...bins.map(b=>Math.max(Number(b.p_ref||0), Number(b.p_cur||0), 1e-9)));
            const x = i => pad + i*(W-2*pad)/bins.length + 6;
            const h = p => (H-2*pad)*Number(p)/maxP;
            ctx.strokeStyle='#9CA3AF'; ctx.beginPath(); ctx.moveTo(pad, H-pad); ctx.lineTo(W-pad, H-pad); ctx.stroke();
            bins.forEach((b,i)=>{ ctx.fillStyle='#E5E7EB'; ctx.fillRect(x(i), H-pad-h(b.p_ref||0), 10, h(b.p_ref||0)); });
            bins.forEach((b,i)=>{ ctx.fillStyle='#93C5FD'; ctx.fillRect(x(i)+12, H-pad-h(b.p_cur||0), 10, h(b.p_cur||0)); });
          }
        }
        // 表格
        const tb = $('mon-psi-table')?.querySelector('tbody');
        if (tb){
          const rows = (res.bins||[]).map((b,i)=>(
            `<tr>
               <td>${i}</td>
               <td class="mono">${fmt(b.lo,3)}</td>
               <td class="mono">${fmt(b.hi,3)}</td>
               <td>${fmt(b.p_ref,4)}</td>
               <td>${fmt(b.p_cur,4)}</td>
               <td>${fmt(b.psi,4)}</td>
             </tr>`
          ));
          tb.innerHTML = rows.length? rows.join('') : `<tr><td colspan="6" class="small">暂无</td></tr>`;
        }
      }catch(e){
        const sum = $('mon-psi-summary');
        if (sum) sum.textContent = '请求失败：' + e.message;
      }
    }

    if (btnAnom) btnAnom.onclick = renderAnom;
    if (btnPSI)  btnPSI.onclick  = renderPSI;

    // 默认跑一次异常扫描
    setTimeout(()=> btnAnom && btnAnom.click(), 500);
  }

  // ---------- 暴露 ----------
  const MonitoringAPI = { anomalyScan, psi, bindDefaultUI };
  global.MonitoringAPI = MonitoringAPI;

})(window);
