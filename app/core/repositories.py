# app/core/repositories.py
"""
【大白话注释】
这个文件就是“数据仓库管理员”。
- 上层（services、adapters、API）只找它拿/存数据，不直接碰时序库或数据库。
- 这样以后把内存TSDB换成 Timescale/Influx，或者把缓存/消息总线换成真实中间件，
  只改底层实现，不动业务代码。

【与真实港口落地的约定】
- 时间统一用 UTC epoch 秒（float）。现场对时 NTP/PTP，API 输入用 ISO8601 时串时，
  请先在接口层转成 UTC 秒再喂给这里。
- 多租户字段 tenant_id 必填，支持一个平台多个码头公司/作业区隔离。
- 资产ID/测点ID的命名建议：全小写、短横线，例如：
  asset_id: "qch-01"（岸桥01）、"y-yd-03"（堆场03），point: "active_power_kw", "status"
- 真实数据来源（落地时只需替换适配器）：
  - 岸桥/场桥/照明/冷站：OPC UA/Modbus（由 adapters/*.py 提供）
  - 充换电/储能PCS：OCPP/Modbus（adapters/*.py）
  - 天气/潮汐/船期：第三方或TOS接口（adapters/tos_client.py 等）
  - 碳因子：文件或API（adapters/carbon_factors.py）
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Literal
import time

# 这里调用“地基层”的时序库门面（上一版我们已新增 app/infra/tsdb.py）
from app.infra.tsdb import TimeSeriesDB

AggType = Literal["raw", "avg", "min", "max", "sum"]
EpochSec = float


# ========= 领域数据结构（尽量贴近真实港口资产与遥测） =========

@dataclass
class Measurement:
    """一条遥测（设备点位的一个时间戳的值）"""
    tenant_id: str         # 多租户：哪个租户/码头公司
    asset_id: str          # 设备ID，例如 'qch-01'（岸桥1）
    point: str             # 点位名，例如 'active_power_kw'
    ts: EpochSec           # UTC秒级时间戳
    value: float           # 数值（如功率kW、温度℃）
    quality: Optional[str] = None  # 可选质量标签，例如 'ok'/'suspect'/'missing_imputed'


@dataclass
class QualityScore:
    """一个时间窗口的数据质量评分（Completeness/Timeliness/Validity）"""
    tenant_id: str
    asset_id: str
    point: str
    window_start: EpochSec
    window_end: EpochSec
    completeness: float     # 0~1
    timeliness: float       # 0~1
    validity: float         # 0~1


@dataclass
class Asset:
    """资产主数据（真实落地时可扩展字段，对应你们资产台账）"""
    tenant_id: str
    asset_id: str
    asset_type: str           # 'quay_crane'/'yard_crane'/'agv'/'lighting'/'chiller'...
    name: Optional[str] = None
    area: Optional[str] = None  # 区域/堆场/泊位等
    vendor: Optional[str] = None
    extra: Optional[dict] = None


# ========= 仓库实现 =========

class TimeSeriesRepo:
    """
    【大白话】时序数据仓库（写点、查曲线、聚合分桶）。
    【谁会调用它】
        - 设备采集适配器（adapters/opcua_client.py、modbus_client.py 等）
        - 业务服务（services/forecast.py、services/energy.py、services/alerts.py）
        - API 层（例如 /api/v1/telemetry/query）
    【它会调用谁】
        - app.infra.tsdb.TimeSeriesDB（当前内存版，后面可换 Timescale/Influx）
    """
    def __init__(self, tsdb: TimeSeriesDB):
        self.tsdb = tsdb

    def write_point(self, m: Measurement) -> None:
        """写入单点（仅把值进TSDB；质量标签单独存）"""
        self.tsdb.write_point(m.asset_id, m.point, m.ts, m.value)

    def write_batch(self, measurements: Iterable[Measurement]) -> None:
        """批量写入；适合采集器每秒/每分钟推一批"""
        self.tsdb.write_batch(
            (m.asset_id, m.point, m.ts, m.value) for m in measurements
        )

    def latest(self, tenant_id: str, asset_id: str, point: str) -> Optional[Tuple[EpochSec, float]]:
        """取最近一个点（后续如加租户级分库，此处会用到 tenant_id 做路由）"""
        return self.tsdb.latest(asset_id, point)

    def query_range(
        self,
        tenant_id: str,
        asset_id: str,
        point: str,
        start: EpochSec,
        end: EpochSec,
        *,
        step_sec: Optional[int] = None,
        agg: AggType = "raw",
    ) -> List[Tuple[EpochSec, float]]:
        """
        查时间窗口：
        - agg='raw'：返回原始点
        - 指定 step_sec 且 agg in {avg,min,max,sum}：返回每个时间桶的聚合值
        例：用于“现在/预测/仿真”曲线里展示历史实绩或回放。
        """
        return self.tsdb.query_range(asset_id, point, start, end, step_sec=step_sec, agg=agg)


class QualityRepo:
    """
    【大白话】质量评分仓库（存 Completeness/Timeliness/Validity）
    先放内存字典，后续换成关系库/时序库的表即可。
    """
    def __init__(self):
        self._q: Dict[tuple, QualityScore] = {}  # key=(tenant_id,asset_id,point,window_start,window_end)

    def write_quality(self, score: QualityScore) -> None:
        key = (score.tenant_id, score.asset_id, score.point, score.window_start, score.window_end)
        self._q[key] = score

    def get_quality(self, tenant_id: str, asset_id: str, point: str, window_start: EpochSec, window_end: EpochSec) -> Optional[QualityScore]:
        return self._q.get((tenant_id, asset_id, point, window_start, window_end))


class AssetRepo:
    """
    【大白话】资产主数据仓库（对应“设备清单与点表”）
    - register_asset：注册/更新资产（真实落地时可以对接CMDB/资产台账）
    - get_asset：读取资产信息（例如UI显示、策略过滤）
    """
    def __init__(self):
        self._assets: Dict[tuple, Asset] = {}  # key=(tenant_id, asset_id)

    def register_asset(self, asset: Asset) -> None:
        self._assets[(asset.tenant_id, asset.asset_id)] = asset

    def get_asset(self, tenant_id: str, asset_id: str) -> Optional[Asset]:
        return self._assets.get((tenant_id, asset_id))


# ========= 为“碳因子库/数据驻留”预留的接口（占位，后续补全到真实实现） =========

class FactorRepo:
    """
    【大白话】碳因子仓库（电网边际/平均、燃料系数等）
    - 这里先占位接口，下一次我会交付 adapters/carbon_factors.py 的真实加载实现，
      并把它接入到这个仓库里，支持地区/版本/时间生效范围查询。
    """
    def __init__(self):
        # key=(region, kind, version) -> kgco2_per_kwh
        self._grid_factors: Dict[tuple, float] = {}

    def upsert_grid_factor(self, region: str, kind: str, version: str, kgco2_per_kwh: float) -> None:
        self._grid_factors[(region, kind, version)] = float(kgco2_per_kwh)

    def get_grid_factor(self, region: str, kind: str, version: Optional[str] = None) -> Optional[float]:
        # 简化：先按 (region,kind,version) 查；没有版本就拿一个最新/默认（后续完善）
        if version:
            return self._grid_factors.get((region, kind, version))
        # 没给版本时，尝试匹配 region+kind 的任意版本（真实实现会按valid_from排序取最新）
        for (r, k, v), val in reversed(list(self._grid_factors.items())):
            if r == region and k == kind:
                return val
        return None


# ========= 帮你快速自测的“烟雾测试” =========

def _demo_smoke_test() -> dict:
    """
    【大白话】本函数是自测工具（不依赖 server），你可以直接运行它看看仓库是否正常工作。
    做的事：
      1）建内存TSDB + 仓库
      2）写入一条 QCH-01 岸桥功率点
      3）查询最近窗口 + 做一次分桶聚合
    """
    tsdb = TimeSeriesDB()
    ts_repo = TimeSeriesRepo(tsdb)
    q_repo = QualityRepo()
    a_repo = AssetRepo()

    now = time.time()
    tenant = "tenant-demo"
    asset = Asset(tenant_id=tenant, asset_id="qch-01", asset_type="quay_crane", name="岸桥01", area="berth-01")
    a_repo.register_asset(asset)

    m = Measurement(tenant_id=tenant, asset_id="qch-01", point="active_power_kw", ts=now, value=123.4, quality="ok")
    ts_repo.write_point(m)

    # 记录一个质量评分（占位）
    q = QualityScore(tenant_id=tenant, asset_id="qch-01", point="active_power_kw",
                     window_start=now-60, window_end=now,
                     completeness=1.0, timeliness=1.0, validity=1.0)
    q_repo.write_quality(q)

    latest = ts_repo.latest(tenant, "qch-01", "active_power_kw")
    window = ts_repo.query_range(tenant, "qch-01", "active_power_kw", start=now-30, end=now+30, step_sec=10, agg="avg")
    return {
        "latest": latest,
        "window_len": len(window),
        "quality": asdict(q_repo.get_quality(tenant, "qch-01", "active_power_kw", now-60, now)),
        "asset": asdict(a_repo.get_asset(tenant, "qch-01")),
    }


if __name__ == "__main__":
    # 允许你直接命令行运行：python -m app.core.repositories
    import json
    print(json.dumps(_demo_smoke_test(), ensure_ascii=False, indent=2))
