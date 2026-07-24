"""
审计与合规导出（Audit Export）· 港口场景友好模拟器
- 端点：由 app/services/opsx/api.py 调用
    POST /api/opsx/audit/make -> make_report()

【大白话】
- 把当前 OpsX 各模块的关键摘要汇总成一份 HTML 报告文件，
  写到 app/static/opsx/ 目录，并返回可访问的 URL（/static/opsx/xxx.html）。
- 真实落地时：
  1) 把“数据源”改为你们的真接口/数据库；
  2) 把“写 HTML”替换为 HTML->PDF（例如 WeasyPrint、wkhtmltopdf），或直接返回你们的对象存储 URL。
"""

from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import os

# ---------- 小工具：安全导入某个模块函数 ----------
def _try_import(path: str, name: str):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def _ensure_static_opsx_dir() -> Path:
    """
    确保 app/static/opsx 目录存在（以本文件为基准向上寻址）
    app/services/opsx/audit_export/simulator.py -> 上溯到 app/ 再 /static/opsx
    """
    here = Path(__file__).resolve()
    # parents: 0=file, 1=audit_export, 2=opsx, 3=services, 4=app
    app_dir = here.parents[3]
    target = app_dir / "static" / "opsx"
    target.mkdir(parents=True, exist_ok=True)
    return target

def _collect_snapshot() -> Dict[str, Any]:
    """
    汇总“当前快照”：尽量从你已经创建的模拟器读取，
    找不到时自动用兜底值，不会报错。
    """
    snap: Dict[str, Any] = {"generated_at": _now()}

    # 1) Rollout
    fn = _try_import("app.services.opsx.rollout_control.simulator", "get_status")
    snap["rollout"] = fn() if fn else {
        "phase":"canary","candidate":"agv_charge@v2.1|yard_crane@v1.8|shore_bess@v1.4",
        "stable":"agv_charge@v2.0|yard_crane@v1.7|shore_bess@v1.3","traffic_pct":0.25
    }

    # 2) Gates
    fn = _try_import("app.services.opsx.quality_gate.simulator", "get_gates")
    snap["gates"] = fn() if fn else {
        "metrics":{"mape":0.032,"guard":0.01,"sla":0.0},
        "thresholds":{"mape_energy_max":0.05,"guard_block_rate_max":0.05,"sla_violation_rate_max":0.02}
    }

    # 3) Profile
    fn = _try_import("app.services.opsx.profile_card.simulator", "get_profile")
    snap["profile"] = fn() if fn else {"values":[0.62,0.55,0.70,0.66,0.40]}

    # 4) Health
    fn = _try_import("app.services.opsx.ops_health.simulator", "get_health")
    snap["health"] = fn() if fn else {
        "score":86,"top_drifts":[
            {"feature":"market_price","psi":0.23},
            {"feature":"active_power","psi":0.18},
            {"feature":"queue_len","psi":0.12}
        ]
    }

    # 5) Timeline（取前 6 条用于摘要）
    fn = _try_import("app.services.opsx.timeline.simulator", "get_timeline")
    tline = fn(horizon_min=60) if fn else {
        "items":[
            {"ts": datetime.utcnow().isoformat(), "kind":"submit", "severity":"info", "text":"提交审批 · 策略 S-341"},
            {"ts": datetime.utcnow().isoformat(), "kind":"abtest", "severity":"warn", "text":"A/B 触发 · ΔkWh 偏差较大"}
        ]
    }
    snap["timeline"] = (tline.get("items") or [])[:6]
    return snap

def _render_html(snap: Dict[str, Any]) -> str:
    """
    用最简单的内联样式渲染一份 HTML（可直接浏览器查看 / 另存为 PDF）
    """
    def esc(x: Any) -> str:
        return (json.dumps(x, ensure_ascii=False, indent=2))

    m = snap.get("gates",{}).get("metrics",{})
    th = snap.get("gates",{}).get("thresholds",{})
    prof = snap.get("profile", {})
    h = snap.get("health", {})
    tl = snap.get("timeline", [])

    def badge(val, thr, good_when="<="):
        # 简单着色：通过绿色，超限红色
        try:
            v = float(val); t = float(thr)
            ok = (v <= t) if good_when == "<=" else (v >= t)
            color = "#16a34a" if ok else "#ef4444"
            return f'<span style="padding:2px 6px;border:1px solid #334155;border-radius:6px;color:{color}">{v:.3f}</span>'
        except Exception:
            return f'<span style="padding:2px 6px;border:1px solid #334155;border-radius:6px;color:#94a3b8">—</span>'

    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>OpsX 作战报告 · {snap.get("generated_at","")}</title>
<style>
  body {{ background:#0b132b; color:#e2e8f0; font-family: ui-sans-serif, system-ui, -apple-system; padding: 24px; }}
  h1,h2,h3 {{ color:#cfe0ff; margin: 0 0 12px; }}
  .card {{ background:#0f172a; border:1px solid #1e2a44; border-radius:12px; padding:14px 16px; margin:12px 0; }}
  .row {{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
  table {{ border-collapse: collapse; width:100%; }}
  th, td {{ border-bottom:1px solid #1e2a44; padding:8px 6px; font-size:14px; }}
  .small {{ color:#94a3b8; font-size:12px; }}
  .pill {{ padding:1px 8px; border:1px solid #334155; border-radius:999px; background:#0b132b; color:#93c5fd; font-size:12px; }}
</style>
</head>
<body>
  <h1>OpsX 作战报告</h1>
  <div class="small">生成时间：{snap.get("generated_at","")} | 本页为示例 HTML，可直接打印为 PDF</div>

  <div class="card">
    <h2>一、上线状态（Rollout）</h2>
    <div class="row">
      <div>阶段：<span class="pill">{snap.get("rollout",{}).get("phase","—")}</span></div>
      <div>候选：<span class="mono">{snap.get("rollout",{}).get("candidate","—")}</span></div>
      <div>稳定：<span class="mono">{snap.get("rollout",{}).get("stable","—")}</span></div>
      <div>流量：<span class="mono">{int(100*(snap.get("rollout",{}).get("traffic_pct") or 0))}%</span></div>
    </div>
  </div>

  <div class="card">
    <h2>二、质量门槛（Gates）</h2>
    <div class="row">
      <div>MAPE {badge(m.get("mape"), th.get("mape_energy_max"), "<=")}</div>
      <div>Guard {badge(m.get("guard"), th.get("guard_block_rate_max"), "<=")}</div>
      <div>SLA {badge(m.get("sla"), th.get("sla_violation_rate_max"), "<=")}</div>
    </div>
    <div class="small mono" style="margin-top:6px;">阈值: {esc(th)}</div>
  </div>

  <div class="card">
    <h2>三、策略画像（摘要）</h2>
    <div class="row">
      <div>五维（动作强度/守护命中/稳定性/收益/风险反向）：</div>
      <div class="mono">{esc(prof.get("values"))}</div>
    </div>
  </div>

  <div class="card">
    <h2>四、运维健康度</h2>
    <div class="row">
      <div>健康分：<span class="pill">{h.get("score","—")}</span></div>
    </div>
    <table style="margin-top:8px;">
      <thead><tr><th>特征</th><th>PSI</th><th>方向</th><th>p50</th><th>ref_p50</th></tr></thead>
      <tbody>
        {''.join(f"<tr><td>{x.get('feature')}</td><td>{x.get('psi')}</td><td>{x.get('direction','')}</td><td>{x.get('p50','')}</td><td>{x.get('ref_p50','')}</td></tr>" for x in (h.get('all') or h.get('top_drifts') or []))}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>五、未来一小时关键事件</h2>
    <table>
      <thead><tr><th>时间</th><th>类型</th><th>级别</th><th>描述</th></tr></thead>
      <tbody>
        {''.join(f"<tr><td class='mono'>{e.get('ts','')}</td><td>{e.get('kind','')}</td><td>{e.get('severity','')}</td><td>{e.get('text','')}</td></tr>" for e in tl)}
      </tbody>
    </table>
  </div>

  <div class="small mono">（注）此报告由 OpsX 模拟器生成。落地时请替换为真实数据与 PDF 导出。</div>
</body></html>
"""
    return html

def make_report() -> Dict[str, Any]:
    """
    生成报告文件并返回 URL
    - 返回: {"url": "/static/opsx/opsx_audit_YYYYmmdd_HHMMSS.html"}
    """
    try:
        snap = _collect_snapshot()
        html = _render_html(snap)

        outdir = _ensure_static_opsx_dir()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"opsx_audit_{ts}.html"
        fpath = outdir / fname
        fpath.write_text(html, encoding="utf-8")

        # 假定 FastAPI/Flask 已经挂载了 /static -> app/static
        url = f"/static/opsx/{fname}"
        return {"url": url}
    except Exception:
        # 写文件失败时，返回一个占位地址（前端至少能点击）
        return {"url": "/download/opsx_audit_demo.pdf"}
