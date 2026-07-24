# app/services/ingest.py
"""
【大白话】
后台采集器（Ingestor）：
- 每 interval_sec 秒，按 step_sec 步长拉取最近一小窗 (window=window_mul*step_sec) 的等间隔序列；
- 来源统一走 di.telemetry.get_series(...)（UTC + 等间隔 + {'ts','v'} 契约）；
- 将 (ts, value) 写入 TimeSeriesRepo/TSDB（仓库层打底，便于后续 KPI/预测/仿真直接复用）；
- 自动跳过重复时间戳，轻量去重；
- 仅依赖 di.telemetry.list_assets() / get_series(...) 两个接口；接真港口时只替换数据源实现。

【真实落地关键口径】
- 时间：UTC，接口接受 epoch 秒，返回 {'ts': ISO8601(UTC), 'v': float}
- 资产与点位：资产类型用于合理边界与清洗；默认采集 active_power_kw，可在 POLL_POINTS 中扩展
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable

# 仓库与时序库
from app.infra.tsdb import TimeSeriesDB
from app.core.repositories import TimeSeriesRepo, Measurement

logger = logging.getLogger("app.ingest")

# 默认采集点配置：不同资产类型采哪些点（可按现场点表扩展）
POLL_POINTS: Dict[str, List[str]] = {
    "quay_crane": ["active_power_kw"],
    "yard_crane": ["active_power_kw"],
    "agv": ["active_power_kw"],
    "lighting": ["active_power_kw"],
    "chiller": ["active_power_kw"],
    "pcs": ["charge_power_kw", "discharge_power_kw"],
    "battery": ["soc"],
    # 未知类型兜底
    "*": ["active_power_kw"],
}

def _to_epoch(ts_iso: str) -> float:
    """ISO8601(UTC) -> epoch 秒"""
    return datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp()

class Ingestor:
    """
    后台采集器（异步任务）：
    - di: 依赖容器（需提供 telemetry.list_assets / telemetry.get_series）
    - interval_sec: 周期
    - step_sec: 等间隔步长（与 get_series 对齐）
    - window_mul: 窗口步数（例如 3 => 3*step_sec）
    - tenant_id: 多租户字段；真实落地按租户注入
    """
    def __init__(
        self,
        di,
        *,
        interval_sec: int = 30,
        step_sec: int = 60,
        window_mul: int = 3,
        tenant_id: str = "tenant-demo",
    ):
        self.di = di
        self.interval_sec = max(1, int(interval_sec))
        self.step_sec = max(1, int(step_sec))
        self.window_mul = max(1, int(window_mul))
        self.tenant_id = tenant_id

        # 统一的 TSDB/Repo 实例（内存版；后续可无缝替换 Timescale/Influx）
        self._tsdb = TimeSeriesDB()
        self.ts_repo = TimeSeriesRepo(self._tsdb)

        # 记忆每个 (asset, point) 已写入的“最近时间戳”，避免重复写
        self._last_written: Dict[Tuple[str, str], float] = {}

        # 任务句柄
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Ingestor started: interval=%ss step=%ss window=%ss",
            self.interval_sec, self.step_sec, self.window_mul * self.step_sec
        )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Ingestor stopped")

    async def _loop(self):
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.exception("Ingest tick failed: %s", e)
            await asyncio.sleep(self.interval_sec)

    async def _tick(self):
        """执行一次采集周期：遍历资产 → 遍历点位 → 拉序列 → 写入仓库（去重）"""
        tel = getattr(self.di, "telemetry", None)
        if tel is None:
            logger.warning("di.telemetry is not available yet")
            return

        # 统一拿资产清单：支持 [{id, type}] 或 [asset_id,...]
        assets = []
        try:
            a = tel.list_assets()
            for it in (a or []):
                if isinstance(it, dict):
                    asset_id = it.get("id") or it.get("asset_id") or it.get("name")
                    a_type = it.get("type") or it.get("asset_type") or "*"
                else:
                    asset_id = str(it)
                    a_type = "*"
                if asset_id:
                    assets.append((asset_id, a_type))
        except Exception:
            logger.exception("list_assets() failed")
            return

        if not assets:
            logger.info("No assets to ingest this tick.")
            return

        now = datetime.now(tz=timezone.utc).timestamp()
        start_ts = now - self.window_mul * self.step_sec
        end_ts = now

        total_points = 0
        writes = 0

        for asset_id, a_type in assets:
            points = POLL_POINTS.get(a_type, POLL_POINTS["*"])
            for point in points:
                try:
                    seq = tel.get_series(asset_id=asset_id, point=point,
                                         start_ts=start_ts, end_ts=end_ts, step_sec=self.step_sec) or []
                except Exception:
                    logger.exception("get_series failed for %s.%s", asset_id, point)
                    continue

                series = []
                for row in seq:
                    try:
                        ts_epoch = _to_epoch(row["ts"]) if isinstance(row.get("ts"), str) else float(row.get("ts"))
                        v = float(row.get("v"))
                        series.append((ts_epoch, v))
                    except Exception:
                        continue

                series.sort(key=lambda x: x[0])
                if not series:
                    continue

                last_key = (asset_id, point)
                last_ts = self._last_written.get(last_key, float("-inf"))

                # 过滤掉已写过的点
                new_points = [(ts, v) for ts, v in series if ts > last_ts]
                total_points += len(series)

                if not new_points:
                    continue

                # 写入仓库（Measurement 契约：tenant/asset/point/ts/value）
                measurements = [
                    Measurement(tenant_id=self.tenant_id, asset_id=asset_id, point=point, ts=ts, value=v, quality="ok")
                    for ts, v in new_points
                ]
                self.ts_repo.write_batch(measurements)
                self._last_written[last_key] = new_points[-1][0]
                writes += len(new_points)

        logger.info("Ingest tick: assets=%d total=%d wrote=%d", len(assets), total_points, writes)


# ============= FastAPI注册入口（给 server.py 调用） =============
def register_ingest_startup(app, di, *, interval_sec: int = 30, step_sec: int = 60, window_mul: int = 3):
    """
    在 FastAPI 中注册后台采集任务（随应用启动/停止）。
    用法（server.py）：
      from app.services.pipeline.ingest import register_ingest_startup
register_ingest_startup(app, di, interval_sec=30, step_sec=60)

    """
    ingestor = Ingestor(di, interval_sec=interval_sec, step_sec=step_sec, window_mul=window_mul)
    app.state.ingestor = ingestor

    async def _start_ingest():
        await app.state.ingestor.start()

    async def _stop_ingest():
        await app.state.ingestor.stop()

    # FastAPI 0.139 moved event registration from the application facade to
    # its router. Keep compatibility with both supported API shapes while the
    # ingestor remains an application-owned lifecycle resource.
    event_target = app if hasattr(app, "add_event_handler") else app.router
    event_target.add_event_handler("startup", _start_ingest)
    event_target.add_event_handler("shutdown", _stop_ingest)


# ============= 额外：不跑 server 的快速自测 =============
async def _self_test(di):
    """手动跑一次 tick（不需要启动 FastAPI），用于 CI/本地自测。"""
    ing = Ingestor(di, interval_sec=1, step_sec=60, window_mul=2)
    await ing._tick()
    return "ok"
