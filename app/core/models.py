# ========================
# app/core/models.py
# ------------------------
# 【文件作用】
# - 定义“功率时间序列里的一个点”（ForecastPoint）和“一组序列的类型别名”（ForecastSeries）。
# - 这是所有服务的“共同语言”，让前后端对齐数据结构。
# ========================

from dataclasses import dataclass           # dataclass 让我们快速定义“只有字段的类”，省掉__init__/__repr__等样板
from typing import List, Dict, Optional     # 类型标注工具：List/Dict/Optional 让阅读更清晰（不用也能跑）

# ↓↓↓ 说明：当前示例暂未用到 Optional，你可以在扩展字段时用它（如 kW: Optional[float] = None 表示可能缺测）

@dataclass
class ForecastPoint:
    """
    【类作用】
    - 表示“时间序列里的一个功率点”。
    - 前端折线图、后端 API 的 points 数组，都由很多个 ForecastPoint 组成。

    【字段含义】
    - ts: ISO8601 字符串时间戳（例如 '2025-09-30T12:00:00Z'），方便前端直接显示/排序；
    - kW: 当时的功率（千瓦）。

    【被谁使用】
    - telemetry_sim：生成最近 60 秒的点（ts+kW）；
    - forecast：生成未来一段时间的点；
    - server：把这些点转成 JSON 返回给前端（常见写法 p.__dict__）。
    """

    ts: str   # 时间戳：页面“最新：xxxx | xx.x kW”里的时间就是它，画横轴也用它
    kW: float # 功率值：页面曲线的纵坐标；右侧 JSON 的每个点里也能看到它

# 类型别名：一个“多设备的序列集合”，形如 {'agv-01': [ForecastPoint,...], 'qc-01': [...]}。
# 说明：这让 forecast/twin 之类的服务可以一次返回多个设备的序列，便于在前端画多条线或做对比。
ForecastSeries = Dict[str, List[ForecastPoint]]
