# ========================
# app/core/ports.py
# ------------------------
# 【文件作用】
# - 定义全项目的“接口契约”（Ports）。谁要提供这些能力，就按这里的函数签名去实现。
# - 上层（路由/孪生等）只面向端口编程，不依赖具体实现 → 便于替换模拟/CSV/现场系统等。
# ========================

from typing import List, Dict, Any                    # 类型标注：List/Dict/Any 用来说明参数/返回的数据结构
from app.core.models import ForecastPoint, ForecastSeries  # 共享模型：功率点 & 多设备序列（设备ID → 点列表）


class TelemetryPort:
    """
    【端口含义】“遥测/执行”能力：读最近功率点、记录/下发动作（演示中是写日志）。
    【谁来实现】app/adapters/telemetry_sim.py（模拟），未来可有 CSV/OPC UA/MQTT/DB 等实现。
    【谁来用它】
      - /api/assets（列设备）通常也放在遥测适配器里；
      - /api/telemetry/recent/{asset} 用它拿“初始 60 个点”；
      - /stream/telemetry/{asset} 每秒拿“最后一个点”；
      - 孪生闭环执行时，用它的“写动作/记日志”接口（真实系统可改为 EMS/SCADA 下发）。
    """

    def get_recent_power(self, asset_id: str) -> List[ForecastPoint]:
        """
        【做什么】给指定设备返回“最近一段时间”的功率点列表（我们约定 60 秒每秒 1 点）。
        【返回值】List[ForecastPoint]，每个点包含 ts（时间）和 kW（功率）。
        【页面对应】左侧折线“初始 60 个点”；以及 SSE 推送用的“末尾点”。
        """
        raise NotImplementedError  # 这是接口；具体逻辑由适配器实现

    def write_actuation_log(self, asset_id: str, action: str, value: float, ts: str) -> None:
        """
        【做什么】把“即将执行/已执行”的动作记录下来（演示：打印/写日志；生产：下发控制）。
        【参数】asset_id=设备，action=动作名（如 charge_power），value=动作值，ts=时间戳。
        【页面对应】点“孪生闭环”后，如果决定执行，后端控制台会看到 [ACTUATION] 日志。
        """
        raise NotImplementedError


class ForecastPort:
    """
    【端口含义】“预测”能力：支持对一组设备做未来窗口的负荷预测。
    【谁来实现】app/services/forecast.py（Simple 版），未来可替换为 ARIMA/LSTM/XGB/图模型。
    【谁来用它】
      - /api/forecast/{asset} 路由；
      - 孪生闭环先预测，再把预测均值作为状态传给 RL。
    """

    def forecast_load(self, asset_ids: List[str], horizon_minutes: int, granularity_minutes: int) -> ForecastSeries:
        """
        【做什么】对一批设备预测未来 horizon_minutes 的负荷，时间步长 granularity_minutes。
        【返回值】ForecastSeries：形如 {'agv-01': [ForecastPoint, ...], 'qc-01': [...]}
        【页面对应】点“预测”按钮后，右侧“输出”里看到的 points（当前前端展示 JSON）。
        """
        raise NotImplementedError


class RLPolicyPort:
    """
    【端口含义】“策略决策（RL/优化器）”能力：根据状态和目标给出动作建议。
    【谁来实现】app/services/rl.py（Stub 版），未来可替换成真正的 PPO/DQN/MILP 混合等。
    【谁来用它】
      - /api/rl/propose 路由（点“RL 策略”按钮）；
      - 孪生闭环调用它拿 proposal，再决定是否执行。
    """

    def propose_actions(self, state: Dict[str, float], objective: str) -> Dict[str, Any]:
        """
        【做什么】输入状态（如 avgForecastKW、soc、price）与目标（cost/carbon/balance），输出建议动作。
        【返回值】Dict：建议 actions、预期影响 expectedImpact、置信度 confidence 等。
        【页面对应】点“RL 策略”后，右侧输出的 JSON。
        """
        raise NotImplementedError


class ReportingPort:
    """
    【端口含义】“报表/核算”能力：计算能耗、碳排等指标，支持不同口径/窗口。
    【谁来实现】app/services/reporting.py（Simple 版），可扩展为多维度/窗口/动态因子。
    【谁来用它】
      - /api/reporting/mini/{asset} 路由（点“报表”按钮）；
      - 未来的定时报表、导出中心也会直接复用。
    """

    def generate_mini_report(self, asset_id: str) -> Dict[str, Any]:
        """
        【做什么】返回一个“简版小报表”（演示口径：最近 1 小时 kWh 与 kgCO2e 估计）。
        【返回值】Dict：例如 {'assetId':..., 'kWh':..., 'kgCO2e':..., 'period':'last_1h_estimate'}
        【页面对应】点“报表”后，右侧输出。
        """
        raise NotImplementedError
