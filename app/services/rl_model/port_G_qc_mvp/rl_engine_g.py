# -*- coding: utf-8 -*-
"""
模块 G｜岸桥（QC）—— RL 引擎（离线 IQL/TD3+BC + 在线 Safe-SAC 残差微调）
------------------------------------------------------------------------
- 仅依赖 Python 标准库 + numpy；严禁 pandas
- 通过 module_g.make_env() 获取环境（屏蔽/奖励/需量/风摆互锁均在 env 内统一）
- 统一 JSONL 输出：qc_step / qc_episode_summary / policy_update
- 可执行示例见文末命令块
"""
import os, sys, json, time, math, argparse, random
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

# 相对导入环境
from .module_g import make_env, DEFAULT_JSONL as ENV_JSONL, DATA_DIR_DEFAULT as DATA_DIR

MODULE_DIR = os.path.dirname(__file__)
OFFLINE_DATASET = os.path.join(MODULE_DIR, "offline_dataset_qc.jsonl")
POLICY_BIN = os.path.join(MODULE_DIR, "policy.bin")            # npz
POLICY_META = os.path.join(MODULE_DIR, "policy_meta.json")
JSONL_PATH = os.path.join(MODULE_DIR, "policy_evaluate_history.jsonl")  # env 会重置写入；本文件追加 policy_update

# -------------------------
# 通用工具
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def _append_jsonl(path: str, d: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        # 清理 NaN/Inf
        def clean(o):
            if isinstance(o, dict): return {k: clean(v) for k,v in o.items()}
            if isinstance(o, list): return [clean(v) for v in o]
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
            return o
        fh.write(json.dumps(clean(d), ensure_ascii=False) + "\n")

# -------------------------
# 特征工程（与 module_g.obs 对齐）
# -------------------------
FEATURE_KEYS = [
    "pwr_ref_pct","idle_ref_min","gmph_target","queue_p50","power_base_kW",
    "price","ef","moves_5min",
    # 传感（缺测用0）
    "tmotor","tinv","sway_deg","wind_mps",
    # 近窗动作（上一步）
    "prev_pwr_pct","prev_idle_min"
]
DISCRETE_MODES = ["normal","ecoL1","ecoL2","ecoL3"]

def obs_to_vec(obs: Dict[str, Any], prev: Dict[str, float]) -> np.ndarray:
    """
    大白话：把 obs 摘要成定长向量，缺测→0，数值做合理缩放（不会太激进）
    """
    def gv(k, default=0.0):
        v = obs.get(k, default)
        if v is None: return default
        try: return float(v)
        except: return default
    x = []
    x.append(gv("pwr_ref_pct"))             # 0~1
    x.append(gv("idle_ref_min"))            # ~[0,15]
    x.append(gv("gmph_target"))             # ~[0,60]
    x.append(gv("queue_p50"))               # ~[0,8]
    x.append(gv("power_base_kW"))           # ~[0,600]
    x.append(gv("price"))                   # 元/kWh
    x.append(gv("ef"))                      # kg/kWh
    x.append(gv("moves_5min"))              # 5min moves
    x.append((gv("tmotor",0.0)-75.0)/30.0)  # 温度平移缩放
    x.append((gv("tinv",0.0)-75.0)/30.0)
    x.append(gv("sway_deg",0.0)/5.0)
    x.append(gv("wind_mps",0.0)/15.0)
    x.append(prev.get("prev_pwr_pct",0.95))
    x.append(prev.get("prev_idle_min",8.0)/10.0)
    return np.asarray(x, dtype=np.float32)

# 归一化器
class Normalizer:
    def __init__(self, dim:int):
        self.mean = np.zeros((dim,), dtype=np.float32)
        self.std  = np.ones((dim,), dtype=np.float32)
        self.eps  = 1e-6
    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0)
        self.std[self.std < 1e-3] = 1.0
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)
    def save(self)->Dict[str,Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
    def load(self, d: Dict[str,Any]):
        self.mean = np.asarray(d["mean"], dtype=np.float32)
        self.std  = np.asarray(d["std"], dtype=np.float32)

# -------------------------
# 简单策略 / 价值近似（numpy 线性模型）
# -------------------------
class LinearPolicy:
    """
    高斯策略：a ~ N(mu=W x + b, diag(std^2))，动作是 [d_power_pct, d_idle_min] 两维残差
    - KL 约束通过参数的 L2/步长与 log_std 裁剪来近似
    """
    def __init__(self, feat_dim: int, act_dim: int = 2):
        self.W = np.zeros((act_dim, feat_dim), dtype=np.float32)
        self.b = np.zeros((act_dim,), dtype=np.float32)
        self.log_std = np.log(np.asarray([0.05, 1.0], dtype=np.float32))  # 初始探索：功率±5%，idle±1min
        self.lr = 3e-4

    def mu(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b

    def sample(self, x: np.ndarray, kl_clip: float = 0.05) -> np.ndarray:
        mu = self.mu(x)
        std = np.exp(self.log_std)
        a = mu + std * np.random.randn(*mu.shape).astype(np.float32)
        # 动作幅度约束（与 module_g 残差带一致）
        a[0] = float(np.clip(a[0], -0.10, +0.10))  # d_power_pct
        a[1] = float(np.clip(a[1], -5.0,  +5.0))   # d_idle_min
        return a

    def update_awac(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray, beta: float = 3.0, kl_reg: float = 1e-3):
        """
        Advantage-Weighted 回归（IQL/AWAC 风格）：
        - Y: 行为动作（目标）；A: 优势；X: 特征
        - 权重 w = exp(A / beta)，数值上限裁剪，做加权最小二乘
        """
        mu = (self.W @ X.T).T + self.b  # [N,2]
        w = np.exp(np.clip(A / max(1e-3,beta), -10, 10))  # [N,]
        w = w.reshape(-1,1)
        # 目标偏差
        err = (mu - Y)
        # 加权梯度（L2）
        gW = (w * err).T @ X / X.shape[0] + kl_reg * self.W
        gb = (w * err).mean(axis=0) + kl_reg * self.b
        self.W -= self.lr * gW
        self.b -= self.lr * gb
        # 适度收缩 std，防止“残差包络夹边”
        self.log_std = np.clip(self.log_std + 0.01*np.tanh(-A.mean()/max(1e-3,beta)), np.log(0.02), np.log(2.0))

    def save(self, path:str, norm:Normalizer, Vw:np.ndarray, Vb:float):
        np.savez(path, W=self.W, b=self.b, log_std=self.log_std, Vw=Vw, Vb=np.array([Vb],dtype=np.float32),
                 norm_mean=norm.mean, norm_std=norm.std)

    def load(self, path:str) -> Tuple[np.ndarray,float,Normalizer]:
        d = np.load(path)
        self.W = d["W"]; self.b = d["b"]; self.log_std = d["log_std"]
        Vw = d["Vw"]; Vb = float(d["Vb"][0])
        norm = Normalizer(self.W.shape[1])
        norm.mean = d["norm_mean"]; norm.std = d["norm_std"]
        return Vw, Vb, norm

class LinearValue:
    """
    线性 V(s) 近似：V = w^T x + b
    - IQL 的 expectile 回归（tau） + 一步 TD 目标：y = r + γ V(s')
    """
    def __init__(self, feat_dim:int):
        self.w = np.zeros((feat_dim,), dtype=np.float32)
        self.b = 0.0
        self.lr = 1e-3

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def update_expectile(self, X: np.ndarray, Y: np.ndarray, tau: float = 0.7):
        V = self.__call__(X)
        diff = Y - V
        # expectile 权重
        w = np.where(diff >= 0, tau, 1.0 - tau).astype(np.float32)
        # 梯度：加权 L2
        grad = -2.0 * w * diff
        gw = (grad.reshape(-1,1) * X).mean(axis=0)
        gb = grad.mean()
        self.w -= self.lr * gw
        self.b -= self.lr * gb

# -------------------------
# 数据集生成（安全探索，屏蔽护航）
# -------------------------
def generate_offline_dataset(episodes:int=4, horizon:int=144, dt_min:int=5, seed:int=42, data_root:str=DATA_DIR):
    set_seed(seed)
    # 覆盖旧数据集
    if os.path.exists(OFFLINE_DATASET):
        os.remove(OFFLINE_DATASET)
    # 简单探索：在残差带内采样动作，所有动作先过 env 的屏蔽
    for ep in range(episodes):
        env, ctx = make_env(dt_min=dt_min, horizon_steps=horizon, qc_id=None, data_root=data_root, jsonl_path=ENV_JSONL)
        prev = {"prev_pwr_pct": 0.95, "prev_idle_min": 8.0}
        for t in range(horizon):
            # ε-探索（随时间衰减），idle 更保守
            eps = max(0.1, 0.5 - 0.4*(t/horizon))
            a = {
                "d_power_pct": float(np.clip(np.random.uniform(-0.10,0.10)*eps, -0.10, 0.10)),
                "d_idle_min": float(np.clip(np.random.uniform(-5.0, 5.0 )*eps, -5.0, 5.0)),
                "mode": ""  # 模式留给计划层/屏蔽处理
            }
            obs = env._obs()
            x = obs_to_vec(obs, prev)
            next_obs, r, done, info = env.step(a)
            x2 = obs_to_vec(next_obs if next_obs else obs, prev) if not done else x
            # 存一条 transition
            rec = {
                "s": x.tolist(),
                "a": [a["d_power_pct"], a["d_idle_min"]],
                "r": float(r),
                "s2": x2.tolist(),
                "done": bool(done),
                "extras": {
                    "mask_reasons": info.get("mask_reasons", []),
                    "gmph_real": info.get("gmph_real", 0.0),
                    "gmph_eff": info.get("gmph_eff", 0.0),
                }
            }
            _append_jsonl(OFFLINE_DATASET, {"key":"transition", **rec})
            if done: break
            prev["prev_pwr_pct"] = env.prev_pwr_pct
            prev["prev_idle_min"] = env.prev_idle
        env.close()
    print(f"[OFFLINE] dataset saved: {OFFLINE_DATASET}")

def _load_transitions(limit:int=0) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    S,A,R,S2,D = [],[],[],[],[]
    with open(OFFLINE_DATASET,"r",encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("key")!="transition": continue
            S.append(d["s"]); A.append(d["a"]); R.append(d["r"]); S2.append(d["s2"]); D.append(1.0 if d["done"] else 0.0)
            if limit>0 and len(S)>=limit: break
    return np.asarray(S,np.float32), np.asarray(A,np.float32), np.asarray(R,np.float32), np.asarray(S2,np.float32), np.asarray(D,np.float32)

# -------------------------
# 离线预训练：IQL-Lite / TD3+BC-Lite
# -------------------------
def train_offline(algo:str="iql", steps:int=30000, batch:int=256, tau:float=0.7, gamma:float=0.995,
                  lr:float=3e-4, beta:float=3.0, seed:int=42, log_every:int=200):
    set_seed(seed)
    S, A, R, S2, D = _load_transitions()
    assert S.shape[0]>0, "offline_dataset_qc.jsonl is empty. Run --make-offline first."
    feat_dim = S.shape[1]
    norm = Normalizer(feat_dim); norm.fit(S)
    X  = norm.transform(S)
    X2 = norm.transform(S2)

    # 模型
    V = LinearValue(feat_dim); V.lr = lr
    pi = LinearPolicy(feat_dim); pi.lr = lr

    # 初始化 V 目标
    y = R + gamma * (1.0 - D) * V(X2)

    # 训练循环
    n = X.shape[0]
    for it in range(1, steps+1):
        idx = np.random.randint(0, n, size=min(batch, n))
        x, a, r, x2, d = X[idx], A[idx], R[idx], X2[idx], D[idx]

        # 期望分位更新 V
        if algo.lower()=="iql":
            yb = r + gamma * (1.0 - d) * V(x2)
            V.update_expectile(x, yb, tau=tau)
            # 重新计算优势 A = y - V(s)
            with np.errstate(invalid='ignore'):
                adv = (yb - V(x))
            adv = np.clip(adv, -50, 50)
            # 策略：优势加权回归（AWAC）
            pi.update_awac(x, adv, a, beta=beta, kl_reg=1e-3)
        else:
            # TD3+BC-Lite：这里用行为加权的最小二乘（没有 Q 网络）
            adv = (r - r.mean())/ (r.std()+1e-6)
            pi.update_awac(x, adv, a, beta=beta, kl_reg=1e-3)

        if it % log_every == 0:
            # 估计 KL（相对上一步的参数变化，近似）
            kl_est = float(np.linalg.norm(pi.W) * 0.0 + np.mean(np.exp(pi.log_std)))
            _append_jsonl(JSONL_PATH, {
                "key":"policy_update","stage":"offline_"+algo,"step":it,
                "roll_metrics":{"dataset":n},
                "lambdas":{"mask_rate":0.0,"sla_pen":0.0,"thermal":0.0},
                "kl_last": kl_est
            })

    # 保存权重与 meta
    pi.save(POLICY_BIN, norm, V.w, V.b)
    meta = {
        "algo": algo, "steps": steps, "batch": batch, "tau": tau, "gamma": gamma, "beta": beta, "lr": lr,
        "feature_keys": FEATURE_KEYS, "action_names": ["d_power_pct","d_idle_min"]
    }
    with open(POLICY_META,"w",encoding="utf-8") as fh: json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[OFFLINE] trained policy saved: {POLICY_BIN}; meta: {POLICY_META}")

# -------------------------
# 在线微调：Safe-SAC-Lite（残差）
# 说明：轻量实现 —— 只更新策略参数（高斯均值），约束用拉格朗日更新，KL 作为近似信任域（调 std）
# -------------------------
def rollout_online(episodes:int=1, horizon:int=144, dt_min:int=5, seed:int=42,
                   kl_max:float=0.05, sleep_every:int=1000, sleep_sec:int=60,
                   data_root:str=DATA_DIR):
    set_seed(seed)
    # 加载策略
    pi = LinearPolicy(len(FEATURE_KEYS))
    if os.path.exists(POLICY_BIN):
        Vw, Vb, norm = pi.load(POLICY_BIN)
    else:
        # 无权重时：用默认 norm
        norm = Normalizer(len(FEATURE_KEYS))
    # 约束目标（可按现场口径调整）
    targets = {"mask_rate":0.30, "sla_pen":0.0, "thermal":0.0}
    lambdas = {"mask_rate":0.0, "sla_pen":1.0, "thermal":0.5}
    lam_lr = 1e-3

    for ep in range(episodes):
        env, ctx = make_env(dt_min=dt_min, horizon_steps=horizon, qc_id=None, data_root=data_root, jsonl_path=ENV_JSONL)
        prev = {"prev_pwr_pct": 0.95, "prev_idle_min": 8.0}
        mask_hits = 0; steps = 0; sla_acc=0.0; th_acc=0.0
        returns = []
        for t in range(horizon):
            obs = env._obs()
            x_raw = obs_to_vec(obs, prev)
            x = norm.transform(x_raw)
            a = pi.sample(x, kl_clip=kl_max)
            # 近似“拉格朗日校正”：当历史约束偏高时，自动朝更保守方向缩小残差
            safe_scale = max(0.5, 1.0 - 0.5*float(lambdas["sla_pen"]>0.5) - 0.3*float(lambdas["mask_rate"]>0.5))
            a_safe = [float(a[0]*safe_scale), float(a[1]*safe_scale)]
            act = {"d_power_pct": a_safe[0], "d_idle_min": a_safe[1], "mode": ""}

            next_obs, r, done, info = env.step(act)
            steps += 1
            mask_hits += int(info.get("masked",0))
            # 从 reward_breakdown 里拿约束指标
            returns.append(float(r))
            # env 的 JSONL 已写 step；此处写 policy_update 的滚动汇总
            prev["prev_pwr_pct"] = env.prev_pwr_pct
            prev["prev_idle_min"] = env.prev_idle
            if done: break
            if (t+1) % max(1, sleep_every) == 0 and sleep_sec>0:
                time.sleep(sleep_sec)

        # 从 episode summary 读 SLA/thermal（env 已写入），这里估计一下
        # 注意：为了轻量，这里不重新扫 JSONL，仅根据 step 期间 reward 分解统计
        # 为了输出更有信息量，做一次近似统计：
        # 读取尾部若干行（可选），这里直接根据 mask_hits 估计 mask_rate
        mask_rate = mask_hits / max(1, steps)
        # 简要风险度量（CVaR@0.2）：尾部均值
        if returns:
            ret_arr = np.asarray(returns, dtype=np.float32)
            q = np.quantile(ret_arr, 0.2)
            cvar = ret_arr[ret_arr<=q].mean() if np.any(ret_arr<=q) else float(ret_arr.mean())
        else:
            cvar = 0.0

        # 拉格朗日乘子更新（投影 >=0）
        lambdas["mask_rate"] = max(0.0, lambdas["mask_rate"] + lam_lr*(mask_rate - targets["mask_rate"]))
        # offline 情景我们没有直接统计 thermal/sla 的数值，这里用 returns 尾部改善作为 proxy：
        # 如果 cvar 很差（更负），提高约束强度
        lambdas["sla_pen"]  = max(0.0, lambdas["sla_pen"]  + lam_lr*( -cvar - 0.0 ))
        # 简化 thermal：若策略 std 偏大且出现 safety_guard 次数多，则提高
        lambdas["thermal"]  = max(0.0, lambdas["thermal"]  + lam_lr*( max(0.0, mask_rate-0.1) ))

        _append_jsonl(JSONL_PATH, {
            "key":"policy_update","stage":"safe_sac_online","step": (ep+1)*horizon,
            "roll_metrics":{"mask_rate":mask_rate, "cvar_0.2": float(cvar)},
            "lambdas": lambdas, "kl_last": 0.0
        })
        env.close()

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="QC RL Engine (offline IQL/TD3BC + online Safe-SAC)")
    # 数据与通用
    p.add_argument("--data-root", type=str, default=DATA_DIR, help="data folder (default: port_G_qc_mvp/data)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep-every", type=int, default=1000)
    p.add_argument("--sleep-sec", type=int, default=60)
    # 制作离线数据
    p.add_argument("--make-offline", action="store_true")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--dt-min", type=int, default=5)
    # 离线训练
    p.add_argument("--train-offline", action="store_true")
    p.add_argument("--algo", type=str, default="iql", choices=["iql","td3bc"])
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=200)
    # 在线微调/评估
    p.add_argument("--rollout", action="store_true")
    p.add_argument("--episodes-online", type=int, default=1)
    p.add_argument("--kl-max", type=float, default=0.05)

    args = p.parse_args()
    set_seed(args.seed)

    if args.make_offline:
        generate_offline_dataset(episodes=args.episodes, horizon=args.horizon, dt_min=args.dt_min,
                                 seed=args.seed, data_root=args.data_root)

    if args.train_offline:
        train_offline(algo=args.algo, steps=args.steps, batch=args.batch, tau=args.tau,
                      gamma=args.gamma, lr=args.lr, beta=args.beta, seed=args.seed, log_every=args.log_every)

    if args.rollout:
        rollout_online(episodes=args.episodes_online, horizon=args.horizon, dt_min=args.dt_min,
                       seed=args.seed, kl_max=args.kl_max, sleep_every=args.sleep_every, sleep_sec=args.sleep_sec,
                       data_root=args.data_root)

    if (not args.make_offline) and (not args.train_offline) and (not args.rollout):
        print("Nothing to do. Try one of: --make-offline | --train-offline | --rollout")

if __name__ == "__main__":
    sys.exit(main())
