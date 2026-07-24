# app/services/portviz/mock_source.py
# ------------------------------------------------------------
# PortViz 模拟数据源（高仿真，可配置，可复现）
#
# 特点：
# - 与 source.PortVizSource 协议对齐（get_bootstrap / next_frame）
# - 支持通过 JSON / 环境变量传入 overrides（见 source.SourceConfig）
# - 支持 seed 固定随机性，方便复现实验
# - 默认几何和 index.html 中 PortViz v5 的 GEO 保持一致
#
# 约定数据结构：
#
# Bootstrap:
#   {
#     "meta": { ... },                    # 可选，来自 overrides["meta"]
#     "world": {"W":1600, "H":900},
#     "lanes": [[{"x":..,"y":..}, ...], ...],
#     "yards": [{"x":..,"y":..,"w":..,"h":..}, ...],
#     "berth": {"x":..,"y":..,"w":..,"h":..},
#     "qcs":   [{"x":..,"y":..}, ...],
#     "ycs":   [{"x":..,"y":..}, ...]
#   }
#
# Frame:
#   {
#     "ts": 1730950000000,                # 毫秒级时间戳
#     "agv":      [{"lane":0,"s":33.2,"alarm":false}, ...],
#     "qc":       [{"busy":true,"trolley":0.42}, ...],
#     "yc":       [{"busy":false}, ...],
#     "tr":       [{"x":820.3,"y":671.2}, ...],
#     "hotspots": [{"x":580,"y":610,"r":48}, ...],
#     "vessels":  [{"berth":0,"progress":0.38,"len":980}, ...]
#   }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import random


@dataclass
class MockSource:
    """
    高仿真模拟源：实现 PortVizSource 协议（见 source.py 的 Protocol）。

    通过 overrides 你可以覆盖：
      - world / lanes / yards / berth / qcs / ycs
      - agv_n / truck_n / hotspots / vessels
      - jitter_agv / jitter_tr / prob_alarm / qc_speed
      - meta（港口画像：名称、代码、吞吐量、时区等）
    """

    overrides: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    # ---------- 构造 ----------
    def __post_init__(self) -> None:
        # 随机源（可复现）
        self._rng = random.Random(self.seed or int(time.time()))

        # -------- 1) 静态几何：世界尺寸 & 基础布局 --------
        world_W = self._ov(("world", "W"), 1600)
        world_H = self._ov(("world", "H"), 900)
        self.world: Dict[str, float] = {"W": float(world_W), "H": float(world_H)}

        # 车道：默认与前端 GEO.lanes 一致，可用 JSON 覆盖
        self.lanes: List[List[Dict[str, float]]] = self._ov(
            "lanes",
            [
                [{"x": 140, "y": 720}, {"x": 1460, "y": 720}],
                [{"x": 140, "y": 640}, {"x": 1460, "y": 640}],
                [{"x": 140, "y": 560}, {"x": 1460, "y": 560}],
                [{"x": 300, "y": 560}, {"x": 300, "y": 720}],
                [{"x": 1280, "y": 560}, {"x": 1280, "y": 720}],
            ],
        )

        # 堆场：如果 JSON 没给，则用规则生成 4x8 的栈场矩阵
        self.yards: List[Dict[str, float]] = self._ov("yards", self._default_yards())

        # 泊位：长条
        self.berth: Dict[str, float] = self._ov(
            "berth",
            {"x": 120, "y": 60, "w": 1360, "h": 24},
        )

        # 岸桥 / 场桥
        self.qcs: List[Dict[str, float]] = self._ov(
            "qcs",
            [{"x": 200 + i * 220, "y": 110} for i in range(6)],
        )
        self.ycs: List[Dict[str, float]] = self._ov(
            "ycs",
            [{"x": 220 + (i % 5) * 260, "y": 320 + (i // 5) * 160} for i in range(10)],
        )

        # meta：完整透传 JSON 里的港口信息（若存在）
        # 例如 port_sgsin_demo.json 中的 port_code/port_name_zh/annual_throughput_teu 等。:contentReference[oaicite:1]{index=1}
        self.meta: Dict[str, Any] = dict(self.overrides.get("meta", {}) or {})

        # -------- 2) 动态实体初始化 --------
        agv_n = int(self.overrides.get("agv_n", 26))
        self.agv: List[Dict[str, Any]] = [
            {
                "lane": i % max(1, min(3, len(self.lanes))),
                "s": (i * 20) % 100,
                "alarm": False,
            }
            for i in range(agv_n)
        ]
        # 速度矢量（可更换分布）
        self._agv_v: List[float] = [10 + (i % 7) for i in range(agv_n)]

        self.qc: List[Dict[str, Any]] = [
            {"busy": (i % 2 == 0), "trolley": self._rng.random()}
            for i in range(len(self.qcs))
        ]
        self.yc: List[Dict[str, Any]] = [
            {"busy": (self._rng.random() < 0.5)} for _ in range(len(self.ycs))
        ]

        tr_n = int(self.overrides.get("truck_n", 12))
        self.tr: List[Dict[str, float]] = [
            {
                "x": 200 + self._rng.random() * 1160,
                "y": 600 + self._rng.random() * 120,
            }
            for _ in range(tr_n)
        ]

        # 拥堵热斑：可覆盖
        self.hotspots: List[Dict[str, float]] = self._ov(
            "hotspots",
            [
                {"x": 580, "y": 610, "r": 48},
                {"x": 900, "y": 660, "r": 44},
                {"x": 1180, "y": 590, "r": 52},
            ],
        )

        # 船舶：沿泊位的“作业进度条”
        self.vessels: List[Dict[str, float]] = self._ov(
            "vessels",
            [
                {"berth": 0, "progress": 0.22, "len": 1000},
                {"berth": 0, "progress": 0.68, "len": 820},
            ],
        )

        # -------- 3) 行为参数（可覆盖） --------
        # 这些字段在 demo JSON 中已经给了默认值。:contentReference[oaicite:2]{index=2}
        self.jitter_agv: float = float(self.overrides.get("jitter_agv", 0.6))
        self.jitter_tr: float = float(self.overrides.get("jitter_tr", 2.0))
        self.prob_alarm: float = float(self.overrides.get("prob_alarm", 0.006))
        self.qc_speed: float = float(self.overrides.get("qc_speed", 0.12))

    # ---------- 工具：读取 overrides ----------
    def _ov(self, key: Any, default: Any) -> Any:
        """
        支持两种索引方式：
        1) 字符串 key（例如 "lanes"）→ overrides["lanes"]
        2) 元组路径（例如 ("world","W")）→ overrides["world"]["W"]
        """
        # 1) 简单 key
        if isinstance(key, str):
            return self.overrides.get(key, default)

        # 2) 路径 key
        cur: Any = self.overrides
        for k in key:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _default_yards(self) -> List[Dict[str, float]]:
        """生成一个 4x8 的规则堆场网格，和前端默认 GEO 保持一致。"""
        yards: List[Dict[str, float]] = []
        for r in range(4):
            for c in range(8):
                yards.append(
                    {
                        "x": 160 + c * 160,
                        "y": 280 + r * 80,
                        "w": 140,
                        "h": 56,
                    }
                )
        return yards

    # ---------- 对外：静态几何 ----------
    def get_bootstrap(self) -> Dict[str, Any]:
        """
        返回前端初始化一次即可的静态几何信息。

        除了 world/lanes/yards/berth/qcs/ycs 以外，如果 meta 存在也会一并透出，
        方便前端在 PortViz 卡片周围展示“港口画像”等文字信息。
        """
        data: Dict[str, Any] = {
            "world": self.world,
            "lanes": self.lanes,
            "yards": self.yards,
            "berth": self.berth,
            "qcs": self.qcs,
            "ycs": self.ycs,
        }
        if self.meta:
            data["meta"] = self.meta
        return data

    # ---------- 对外：动态一帧 ----------
    def next_frame(self, since: Optional[int] = None) -> Dict[str, Any]:
        """
        生成一帧动态数据。

        参数:
        - since: 上一帧的时间戳（毫秒）。目前仅用于时间对齐，未做基于 dt 的物理建模，
          前端通过轮询频率和 speed 控制整体节奏。
        """
        ts = int(time.time() * 1000)

        # AGV：在各自车道上前进 + 轻微抖动 + 偶发告警
        agv_out: List[Dict[str, Any]] = []
        for i, a in enumerate(self.agv):
            v = self._agv_v[i % len(self._agv_v)]
            # 匀速推进 + 抖动
            a["s"] = (
                a["s"]
                + v * 0.04
                + (self._rng.random() * self.jitter_agv - self.jitter_agv / 2)
            ) % 100
            # 偶发告警（短闪）
            if not a["alarm"] and self._rng.random() < self.prob_alarm:
                a["alarm"] = True
            elif a["alarm"] and self._rng.random() < 0.25:
                a["alarm"] = False

            agv_out.append(
                {
                    "lane": int(a["lane"]),
                    "s": float(a["s"]),
                    "alarm": bool(a["alarm"]),
                }
            )

        # QC：小车往返 + 忙闲切换
        qc_out: List[Dict[str, Any]] = []
        for q in self.qc:
            q["trolley"] = (q["trolley"] + self._rng.random() * self.qc_speed) % 1.0
            if self._rng.random() < 0.10:
                q["busy"] = not q["busy"]
            qc_out.append(
                {
                    "busy": bool(q["busy"]),
                    "trolley": float(q["trolley"]),
                }
            )

        # YC：忙闲轻微变化
        yc_out: List[Dict[str, Any]] = []
        for y in self.yc:
            if self._rng.random() < 0.05:
                y["busy"] = not y["busy"]
            yc_out.append({"busy": bool(y["busy"])})

        # 拖车：在南侧车道附近游走
        tr_out: List[Dict[str, Any]] = []
        for t in self.tr:
            t["x"] = max(
                180,
                min(
                    1420,
                    t["x"] + (self._rng.random() * self.jitter_tr - self.jitter_tr / 2),
                ),
            )
            t["y"] = max(
                520,
                min(780, t["y"] + (self._rng.random() * 1.6 - 0.8)),
            )
            tr_out.append({"x": float(t["x"]), "y": float(t["y"])})

        # 船舶：沿泊位长度缓慢推进
        vessels_out: List[Dict[str, Any]] = []
        for v in self.vessels:
            pr = (v["progress"] + 0.0008) % 1.0
            v["progress"] = pr
            vessels_out.append(
                {
                    "berth": int(v["berth"]),
                    "progress": float(pr),
                    "len": float(v["len"]),
                }
            )

        return {
            "ts": ts,
            "agv": agv_out,
            "qc": qc_out,
            "yc": yc_out,
            "tr": tr_out,
            "hotspots": list(self.hotspots),
            "vessels": vessels_out,
        }
