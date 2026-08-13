from __future__ import annotations

import re
from typing import Any, Dict, List

from .service import CurvesService


GROUPS = ["QC", "YC", "AGV", "BESS", "REEFER", "LIGHT", "HVAC", "SHORE", "GATE", "OTHER"]


class CurvesStacked:
    """Sum the actual asset-level backend series into operational groups.

    No group shape or share is generated locally.  Every visible value comes
    from calibrated telemetry (now), the fitted forecast model (forecast), or
    the selected hash-verified policy runtime (sim).  This makes the group sum
    auditable and exactly mass-conserving with the asset-level curves.
    """

    def __init__(self, di: Any) -> None:
        self.di = di
        self.curves = CurvesService(di)

    @staticmethod
    def _group_of(asset: Dict[str, Any]) -> str:
        text = " ".join(
            str(asset.get(key) or "")
            for key in ("type", "asset_type", "category", "name", "label", "id", "asset_id")
        ).lower()
        patterns = (
            ("QC", r"\b(qc|quay crane|sts)\b|岸桥|桥吊"),
            ("YC", r"\b(yc|rtg|rmg|yard crane)\b|场桥"),
            ("AGV", r"\b(agv|truck|terminal tractor)\b|集卡|拖车"),
            ("BESS", r"\b(bess|battery|ess)\b|储能"),
            ("REEFER", r"\breefer\b|冷藏箱"),
            ("LIGHT", r"\b(light|lighting)\b|照明|高杆灯"),
            ("HVAC", r"\b(hvac|chiller)\b|冷站|空调|制冷"),
            ("SHORE", r"shore[ -]?power|cold ironing|岸电"),
            ("GATE", r"\bgate\b|闸口|铁水联运"),
        )
        for group, pattern in patterns:
            if re.search(pattern, text):
                return group
        return "OTHER"

    def _assets(self, limit: int) -> List[Dict[str, Any]]:
        try:
            rows = self.di.telemetry.list_assets() or []
        except Exception:
            return []
        output: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            asset_id = row.get("id") or row.get("asset_id")
            if not asset_id or not row.get("include_in_aggregate", True):
                continue
            output.append({"id": str(asset_id), "group": self._group_of(row)})
            if len(output) >= limit:
                break
        return output

    def stacked_power(
        self,
        mode: str = "forecast",
        horizon_min: int = 360,
        step_min: int = 1,
        limit: int = 200,
    ) -> Dict[str, Any]:
        assets = self._assets(limit)
        rows: List[tuple[str, str, List[Dict[str, Any]], str]] = []
        for asset in assets:
            payload = self.curves.asset(
                asset["id"],
                mode=mode,
                horizon_min=horizon_min,
                step_min=step_min,
                scenario="strategy" if mode == "sim" else "baseline",
            )
            points = ((payload.get("series") or {}).get("p50") or []) if payload.get("available") else []
            if points:
                rows.append((asset["id"], asset["group"], points, str(payload.get("_source") or "unknown")))

        length = min((len(points) for _, _, points, _ in rows), default=0)
        empty_series = {group: [] for group in GROUPS}
        if not length:
            return {
                "mode": mode,
                "available": False,
                "unit": "kW",
                "groups": GROUPS,
                "x": [],
                "series": empty_series,
                "total": [],
                "reason": "no asset-level backend series",
            }

        aligned = [(asset_id, group, points[-length:], source) for asset_id, group, points, source in rows]
        x = [point.get("ts") for point in aligned[0][2]]
        series: Dict[str, List[float]] = {group: [0.0] * length for group in GROUPS}
        group_counts = {group: 0 for group in GROUPS}
        for _asset_id, group, points, _source in aligned:
            group_counts[group] += 1
            for index, point in enumerate(points):
                series[group][index] += float(point.get("kW", 0.0) or 0.0)
        series = {
            group: [round(value, 6) for value in values]
            for group, values in series.items()
        }
        total = [round(sum(series[group][index] for group in GROUPS), 6) for index in range(length)]
        return {
            "mode": mode,
            "available": True,
            "unit": "kW",
            "groups": GROUPS,
            "x": x,
            "series": series,
            "total": total,
            "group_counts": group_counts,
            "asset_count": len(aligned),
            "_source": "sum_of_asset_level_backend_series",
            "upstream_sources": sorted({source for _, _, _, source in aligned}),
        }
