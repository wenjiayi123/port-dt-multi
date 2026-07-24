from __future__ import annotations

def enforce(action: dict, constraints: dict) -> dict:
    """
    简化版运行时安全盾：检查容量、爬坡、热边界。返回是否通过及可能的截断动作。
    action: {"power": kW, "prev_power": kW, "temp": C}
    constraints: {"cap": kW, "ramp": kW/s, "temp_max": C}
    """
    power = float(action.get("power", 0.0))
    prev = float(action.get("prev_power", power))
    temp = float(action.get("temp", 25.0))
    cap = float(constraints.get("cap", 5000.0))
    ramp = float(constraints.get("ramp", 500.0))
    tmax = float(constraints.get("temp_max", 85.0))

    ok = True
    reasons = []

    # 容量
    if abs(power) > cap:
        ok = False
        reasons.append("capacity_limit")
        power = max(min(power, cap), -cap)

    # 爬坡
    if abs(power - prev) > ramp:
        ok = False
        reasons.append("ramp_rate")
        if power > prev:
            power = prev + ramp
        else:
            power = prev - ramp

    # 热约束
    if temp > tmax:
        ok = False
        reasons.append("thermal_bound")
        power = 0.0  # 强行降载

    return {"ok": ok, "power": power, "reasons": reasons}
