# app/infra/tsdb.py
"""
轻量级时序库（TSDB）门面：先用内存实现，后续可无缝替换 Timescale/Influx。
- write_point / write_batch：写入数据点
- query_range：按时间范围取数据，支持分桶聚合（avg/min/max/sum）
- latest：取最近一个点
- retention_hours：数据保留时长（内存版定期修剪）

后续只让上层通过 Repository 调用，不直接依赖具体存储，便于后换库。
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
import time
import math
import bisect
from collections import defaultdict

Number = float
EpochSec = float
Key = tuple[str, str]  # (asset_id, point)

__all__ = ["TimeSeriesDB", "AggFn"]


class AggFn:
    """聚合函数集合。"""
    @staticmethod
    def avg(values: List[Number]) -> Optional[Number]:
        return sum(values) / len(values) if values else None

    @staticmethod
    def min(values: List[Number]) -> Optional[Number]:
        return min(values) if values else None

    @staticmethod
    def max(values: List[Number]) -> Optional[Number]:
        return max(values) if values else None

    @staticmethod
    def sum(values: List[Number]) -> Optional[Number]:
        return sum(values) if values else None


class TimeSeriesDB:
    """
    极简内存版 TSDB：
    - _store: dict[(asset_id, point)] -> [(ts, value)]（按 ts 递增）
    - retention_hours：超过保留期的数据会被修剪
    """
    def __init__(self, retention_hours: int = 24 * 7):
        self._store: Dict[Key, List[Tuple[EpochSec, Number]]] = defaultdict(list)
        self.retention_sec = retention_hours * 3600

    # ---------- 写入 ----------
    def write_point(self, asset_id: str, point: str, ts: EpochSec, value: Number) -> None:
        """写入单点，保持时间有序。"""
        key = (asset_id, point)
        series = self._store[key]
        # 二分插入保持有序
        idx = bisect.bisect_left(series, (ts, -math.inf))
        if idx < len(series) and series[idx][0] == ts:
            series[idx] = (ts, float(value))  # 同一时间戳覆盖
        else:
            series.insert(idx, (ts, float(value)))
        self._prune(key)

    def write_batch(self, measurements: Iterable[Tuple[str, str, EpochSec, Number]]) -> None:
        """批量写入 [(asset_id, point, ts, value), ...]"""
        for asset_id, point, ts, val in measurements:
            self.write_point(asset_id, point, ts, val)

    # ---------- 查询 ----------
    def latest(self, asset_id: str, point: str) -> Optional[Tuple[EpochSec, Number]]:
        series = self._store.get((asset_id, point), [])
        return series[-1] if series else None

    def query_range(
        self,
        asset_id: str,
        point: str,
        start: EpochSec,
        end: EpochSec,
        *,
        step_sec: Optional[int] = None,
        agg: str = "raw",
    ) -> List[Tuple[EpochSec, Number]]:
        """
        查询时间窗口数据。
        - agg='raw'：返回原始点
        - 指定 step_sec 且 agg in {avg,min,max,sum}：进行分桶聚合并返回每个桶中心时间戳
        """
        series = self._store.get((asset_id, point), [])
        if not series:
            return []

        # 截取窗口
        left = bisect.bisect_left(series, (start, -math.inf))
        right = bisect.bisect_right(series, (end, math.inf))
        window = series[left:right]

        if agg == "raw" or step_sec is None:
            return window

        agg_fn = getattr(AggFn, agg, None)
        if not callable(agg_fn):
            raise ValueError(f"unsupported agg: {agg}")

        if step_sec <= 0:
            raise ValueError("step_sec must be positive")

        # 分桶： [start, start+step) ,[start+step, start+2*step) , ...
        buckets: List[List[Number]] = []
        bucket_ts: List[EpochSec] = []
        n_buckets = int(max(1, math.ceil((end - start) / step_sec)))
        buckets = [[] for _ in range(n_buckets)]
        bucket_ts = [start + (i + 0.5) * step_sec for i in range(n_buckets)]

        for ts, val in window:
            idx = int((ts - start) // step_sec)
            if 0 <= idx < n_buckets:
                buckets[idx].append(val)

        out: List[Tuple[EpochSec, Number]] = []
        for t, vals in zip(bucket_ts, buckets):
            v = agg_fn(vals)
            if v is not None:
                out.append((t, float(v)))
        return out

    # ---------- 维护 ----------
    def _prune(self, key: Key) -> None:
        """按保留期修剪过老数据。"""
        if self.retention_sec <= 0:
            return
        series = self._store.get(key, [])
        if not series:
            return
        cutoff = time.time() - self.retention_sec
        idx = bisect.bisect_left(series, (cutoff, -math.inf))
        if idx > 0:
            del series[:idx]
