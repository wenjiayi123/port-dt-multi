# app/infra/message_bus.py
"""
【大白话注释】
这是“统一消息总线”的门面。现在是内存版 pub/sub，支持 MQTT 风格通配：
- 'foo/+/bar' 里的 '+' 匹配一个片段
- 'foo/#' 里的 '#' 匹配后续所有片段
主题用点号或斜杠都可以（内部统一成斜杠比较好），示例：
  telemetry/qch-01/active_power_kw
  strategy/execute/qch-01
这样以后把 backend_url 改成 kafka:// 或 mqtt:// 就能换后端。

【建议主题命名规范】
- 遥测：telemetry/{asset_id}/{point}
- 告警：alarm/{asset_id}/{point}  或 alarm/{level}
- 策略：strategy/propose ；strategy/execute/{policy_id}
- 审计：audit/{event_id}

【线程/协程说明】
本内存实现是同步回调，轻量自测用。生产替换为 Kafka/MQTT 后，这里会变成异步收发。

【谁会调用它】
- di.bus.publish(...) / di.bus.subscribe(...)（我下一步会在 di.py 注册为 di.bus）
- 采集适配器、预测/告警/策略服务、SSE 推送器

【替换后端】
- backend_url = "memory://"（默认）
- 未来可支持： "kafka://broker-1,broker-2?topic_prefix=..." ； "mqtt://host:1883?user=...&password=..."
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple
import threading
import re
from dataclasses import dataclass
from time import time

__all__ = ["MessageBus", "Message"]

@dataclass
class Message:
    topic: str
    payload: dict
    ts: float

def _topic_to_path(topic: str) -> str:
    # 统一把 . 替换成 /，便于通配匹配：a.b.c -> a/b/c
    return str(topic).replace(".", "/").strip("/")

def _match(pattern: str, topic: str) -> bool:
    """
    MQTT 风格通配：
      '+' 匹配单层
      '#' 匹配多层（含空）
    """
    p = _topic_to_path(pattern).split("/")
    t = _topic_to_path(topic).split("/")
    i = j = 0
    while i < len(p) and j < len(t):
        if p[i] == "#":
            # '#' 吃掉后续所有
            return True
        if p[i] == "+":
            i += 1; j += 1
            continue
        if p[i] != t[j]:
            return False
        i += 1; j += 1
    # 末尾处理
    if i == len(p) and j == len(t):
        return True
    if i < len(p) and p[i] == "#":
        return True
    return False

class MessageBus:
    """
    统一消息总线（内存实现）：
    - publish(topic, payload)
    - subscribe(pattern, callback)
    支持 MQTT 通配，callback 原型：fn(msg: Message) -> None
    """
    def __init__(self, backend_url: str = "memory://"):
        self.backend_url = backend_url
        self._subs: List[Tuple[str, Callable[[Message], None]]] = []
        self._lock = threading.Lock()

    def subscribe(self, pattern: str, callback: Callable[[Message], None]) -> None:
        """订阅主题（支持 +/# 通配）。"""
        with self._lock:
            self._subs.append((pattern, callback))

    def publish(self, topic: str, payload: dict) -> int:
        """发布消息，返回实际投递的订阅者数量。"""
        topic_norm = _topic_to_path(topic)
        msg = Message(topic=topic_norm, payload=dict(payload or {}), ts=time())
        delivered = 0
        with self._lock:
            subs = list(self._subs)
        for pat, cb in subs:
            if _match(pat, topic_norm):
                try:
                    cb(msg)
                    delivered += 1
                except Exception:
                    # 生产可接 Sentry/日志；此处不抛出，避免影响其它订阅者
                    pass
        return delivered

# ========== 冒烟测试 ==========
def _smoke() -> dict:
    """
    演示：
      1) 订阅 telemetry/+/# （所有设备的所有点）
      2) 发布一条 telemetry.qch-01.active_power_kw
    预期：可收到 1 条消息。
    """
    bus = MessageBus()
    received = []

    def on_msg(m: Message):
        received.append({"topic": m.topic, "payload": m.payload})

    bus.subscribe("telemetry/+/+", on_msg)
    cnt = bus.publish("telemetry.qch-01.active_power_kw", {"v": 123.4})
    return {"delivered": cnt, "received_len": len(received), "sample": received[:1]}

if __name__ == "__main__":
    import json
    print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
