
class Adapter:
    """场桥待机/功率：生成队列/负载/功率；接受 mode: eco|run|idle"""
    def __init__(self, cfg=None):
        import random
        self.cfg = cfg or {}; self.dt = int(self.cfg.get("step_sec", 600)); self.t = 0
        random.seed(self.cfg.get("sim", {}).get("seed", 31)); self.load = 0.6; self.idle = 0.2
    def reset(self): return self.read_state()
    def step(self, action=None): return self.command(action or {})
    def read_state(self):
        import random as _r
        power_kw = 50 + 200*self.load + 20*self.idle
        qlen = int(10*self.load + _r.uniform(-2,2))
        return {"time": self.t, "queue_len": qlen, "load_ratio": self.load, "idle_ratio": self.idle, "power_kw": power_kw}
    def command(self, payload):
        mode = (payload or {}).get("mode")
        if mode == "eco": self.load = max(0.0, self.load - 0.05)
        elif mode == "run": self.load = min(1.0, self.load + 0.05)
        else:
            import random as _r
            self.idle = min(1.0, max(0.0, self.idle + _r.uniform(-0.05,0.05)))
        self.t += self.dt; return self.read_state()
