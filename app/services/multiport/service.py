# app/services/multiport/service.py
# 功能：向 /api/multiport/summary 提供汇总数据
# 说明：仅读取带可审计来源说明的快照；不生成跨港收益或保真度样例。

from __future__ import annotations

from pathlib import Path  # [1]
import json  # [3]
from typing import Any, Dict, List  # [4]
import os  # [3a]

class MultiportService:  # [6]
    """
    提供“多港口 / 多场景一体化管理”的汇总数据。
    返回结构：
    {
      "updated_at": "2025-12-04T08:00:00Z",
      "ports": [
        {
          "id": "port-a",
          "name": "Port A · 集装箱港",
          "phase": "PoC|Pilot|全量",
          "twin_fidelity": 0.93,
          "annual_saving_mwy": 1200,
          "annual_co2_t": 3800,
          "scenes": ["AGV 充电节能调度", ...]
        },
        ...
      ]
    }
    """  # [22]

    def __init__(self) -> None:  # [24]
        # Optional external multi-port data path
        # app/services/multiport/data/summary_snapshot.json
        default_path = Path(__file__).with_name("data") / "summary_snapshot.json"  # [27]
        env_path = os.getenv("PORT_MULTI_PORT_SNAPSHOT", "").strip()
        self._data_file = Path(env_path) if env_path else default_path

    def get_summary(self) -> Dict[str, Any]:  # [29]
        """
        返回汇总数据。文件需内联 ``_provenance`` 或同名
        ``.meta.json``，且明确来源 URL 和证据类型。
        """  # [33]
        data = None  # [34]

        # Prefer the external data file when configured.
        try:  # [37]
            if self._data_file.exists():  # [38]
                with self._data_file.open("r", encoding="utf-8") as f:  # [39]
                    data = json.load(f)  # [40]
        except Exception:
            data = None  # [42]

        if not isinstance(data, dict) or "ports" not in data:
            return {"available": False, "updated_at": None, "ports": [], "reason": "multi-port snapshot is missing or invalid"}

        metadata = data.get("_provenance")
        sidecar = self._data_file.with_suffix(".meta.json")
        if metadata is None and sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                metadata = None
        if not isinstance(metadata, dict) or metadata.get("provenance_type") not in {"public", "port_export", "audited"} or not metadata.get("source_url"):
            return {
                "available": False,
                "updated_at": None,
                "ports": [],
                "reason": "multi-port snapshot lacks auditable provenance metadata",
            }

        # 3) 轻度校验与规范化
        data = self._normalize(data)  # [49]
        data["available"] = True
        data["_provenance"] = metadata
        return data  # [50]

    # ----------------- 内部实现 -----------------

    def _normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:  # [54]
        """填补缺省字段、格式化时间、限制取值范围等。"""  # [55]
        ports: List[Dict[str, Any]] = list(data.get("ports") or [])  # [56]

        # 统一时间
        ua = data.get("updated_at")  # [59]
        data["updated_at"] = ua  # [62]

        # 端口项规范化
        for p in ports:  # [65]
            p.setdefault("id", "unknown")  # [66]
            p.setdefault("name", p["id"])  # [67]
            p.setdefault("phase", None)  # [68]
            p.setdefault("twin_fidelity", None)  # [69]
            p.setdefault("annual_saving_mwy", None)  # [70]
            p.setdefault("annual_co2_t", None)  # [71]
            p.setdefault("scenes", [])  # [72]

            # twin_fidelity 限制到 [0,1]
            try:  # [75]
                tf = float(p["twin_fidelity"]) if p.get("twin_fidelity") is not None else None
                if tf is not None:
                    if tf < 0:
                        tf = 0.0
                    if tf > 1:
                        tf = 1.0
                    p["twin_fidelity"] = tf  # [81]
            except Exception:
                p["twin_fidelity"] = None  # [83]

            # 数值字段转为数值
            for key in ("annual_saving_mwy", "annual_co2_t"):  # [86]
                try:
                    if p.get(key) is not None:
                        p[key] = float(p[key]) if key == "annual_co2_t" else int(p[key])
                except Exception:
                    p[key] = None  # [90]

            # scenes 统一成字符串列表
            scenes = p.get("scenes", [])  # [93]
            if not isinstance(scenes, list):  # [94]
                scenes = [str(scenes)]
            p["scenes"] = [str(s) for s in scenes]  # [96]

        data["ports"] = ports  # [98]
        return data  # [99]
