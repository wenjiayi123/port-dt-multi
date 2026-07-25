# ============================================
# app/services/rl_train.py
# --------------------------------------------
# 离线训练壳 + 策略产物落盘 + 自动发起影子评估
# CEM (Cross-Entropy Method) policy search
#
# 大白话：
#   - 不依赖深度学习大库；CEM 是黑盒策略优化方法，比随机搜索更稳定、更快收敛；
#   - 训练对象是 PolicyParams（对动作做比例/限幅/偏置），与 rl.py 的运行时加载完全一致；
#   - 评估环境用 PortEnergyEnvPro（与面板仿真一致），奖励为 env 的 reward；
#   - 产物落到 data/objects/rl/policies/policy-*.json；可选自动登记候选并启动影子。
# ============================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse, json, math, os, random, time

# --------- 依赖（缺失时报清晰错误） ----------
try:
    from app.services.rl_suite.rl_env_pro import PortEnergyEnvPro
except Exception as e:
    PortEnergyEnvPro = None

try:
    from app.services.rl_suite.rl_rollout import RLPolicyRollout
except Exception:
    RLPolicyRollout = None


# --------- 小工具 ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _read_json(p: str) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def _write_json(p: str, obj: Any) -> None:
    _ensure_dir(str(Path(p).parent))
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --------- 数据集 IO（兼容合成与真实） ----------
@dataclass
class TrajStep:
    t: int
    obs: Dict[str, Any]
    acts: List[Dict[str, Any]]
    rew: float
    info: Dict[str, Any]

@dataclass
class Trajectory:
    created_at: str
    episodes: int
    steps: List[TrajStep]


def load_dataset(dir_or_files: List[str] | str) -> List[Trajectory]:
    """
    加载离线轨迹：
      - 支持传文件夹（自动扫描 *.json）或文件列表；
      - 兼容 rl_env_pro.rollouts_pro() 生成的 schema。
    """
    files: List[str] = []
    if isinstance(dir_or_files, str):
        p = Path(dir_or_files)
        if p.is_dir():
            files = [str(x) for x in sorted(p.glob("*.json"))]
        else:
            files = [str(p)]
    else:
        files = list(dir_or_files)

    trajs: List[Trajectory] = []
    for f in files:
        try:
            data = _read_json(f)
            steps = []
            for s in data.get("steps", []):
                steps.append(TrajStep(
                    t=int(s.get("t", 0)),
                    obs=dict(s.get("obs", {})),
                    acts=list(s.get("acts", [])),
                    rew=float(s.get("rew", 0.0)),
                    info=dict(s.get("info", {})),
                ))
            trajs.append(Trajectory(
                created_at=data.get("created_at", _now_iso()),
                episodes=int(data.get("episodes", 0)),
                steps=steps
            ))
        except Exception:
            # 跳过损坏文件
            continue
    return trajs


# --------- 策略参数（与 rl.py / PolicyParams.apply 对齐） ----------
@dataclass
class PolicyParams:
    """对默认动作的“比例/限幅/偏置”参数（训练目标）"""
    bess_charge_scale: float = 1.0
    bess_discharge_scale: float = 1.0
    agv_charge_scale: float = 1.0
    lighting_reduce_scale: float = 1.0
    chiller_delta_scale: float = 1.0
    max_kw_cap: float = 1000.0  # 动作限幅（防返回离谱值）

    def as_vector(self) -> List[float]:
        # 便于 CEM 进行高斯采样
        return [
            self.bess_charge_scale,
            self.bess_discharge_scale,
            self.agv_charge_scale,
            self.lighting_reduce_scale,
            self.chiller_delta_scale,
            self.max_kw_cap,
        ]

    @staticmethod
    def from_vector(v: List[float]) -> "PolicyParams":
        # 各维的安全边界（防止采样出不合理值）
        def clamp(x, lo, hi): return max(lo, min(hi, x))
        return PolicyParams(
            bess_charge_scale=clamp(float(v[0]), 0.3, 1.8),
            bess_discharge_scale=clamp(float(v[1]), 0.3, 1.8),
            agv_charge_scale=clamp(float(v[2]), 0.3, 1.8),
            lighting_reduce_scale=clamp(float(v[3]), 0.3, 1.8),
            chiller_delta_scale=clamp(float(v[4]), 0.5, 1.5),
            max_kw_cap=clamp(float(v[5]), 100.0, 1500.0),
        )

    def apply(self, acts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for a in acts:
            a = dict(a)
            cmd = str(a.get("cmd"))
            if "kW" in a:
                kw = float(a.get("kW", 0.0))
                if cmd == "charge" and a.get("asset") and "bess" in a["asset"]:
                    kw *= self.bess_charge_scale
                elif cmd == "discharge" and a.get("asset") and "bess" in a["asset"]:
                    kw *= self.bess_discharge_scale
                elif cmd == "charge" and a.get("asset") == "agv-fleet":
                    kw *= self.agv_charge_scale
                elif cmd == "reduce":
                    kw *= self.lighting_reduce_scale
                kw = max(0.0, min(abs(kw), self.max_kw_cap))
                a["kW"] = kw
            if cmd == "set_sp_delta":
                delta = float(a.get("delta_c", 0.0)) * self.chiller_delta_scale
                a["delta_c"] = max(-2.0, min(2.0, delta))
            out.append(a)
        return out


# --------- 训练器：CEM + 随机搜索（兼容原 cql/bcq 占位） ---------
class OfflineRLTrainer:
    """
    大白话：
      - CEM：对 PolicyParams 的 6 维向量进行高斯族采样 → 选精英 → 更新均值/方差（带平滑与剪裁）；
      - evaluate_params()：使用 Pro 环境随机场景评估平均回报；
      - 兼容旧接口：train_cql/train_bcq 仍然存在（默认委托 CEM 或随机搜索）。
    """

    def __init__(self, env: Optional[PortEnergyEnvPro] = None, seed: int = 42):
        if PortEnergyEnvPro is None:
            raise RuntimeError("缺少 rl_env_pro.PortEnergyEnvPro，请先添加 app/services/rl_env_pro.py")
        self.env = env or PortEnergyEnvPro()
        random.seed(seed)

    # ---- 评估某组参数在环境上的期望回报（使用 Pro 环境） ----
    def evaluate_params(self, params: PolicyParams, episodes: int = 3) -> Dict[str, Any]:
        total = 0.0
        ep_details: List[Dict[str, Any]] = []
        for ep in range(episodes):
            obs = self.env.reset(self.env._random_ctx())  # 随机场景
            done = False
            ep_ret = 0.0
            while not done:
                base_acts = self.env._default_policy_step()   # Replaceable behavior-policy baseline
                acts = params.apply(base_acts)                # 应用训练参数
                obs, rew, done, info = self.env.step(acts)
                ep_ret += float(rew)
            total += ep_ret
            ep_details.append({"episode": ep, "return": ep_ret})
        avg = total / max(1, episodes)
        return {"avg_return": avg, "details": ep_details}

    # ---- CEM 策略搜索 ----
    def train_cem(
        self,
        iters: int = 25,
        population: int = 32,
        elite_frac: float = 0.2,
        episodes_per_eval: int = 3,
        init_mean: Optional[PolicyParams] = None,
        init_sigma: Optional[List[float]] = None,
        smoothing: float = 0.8,
        sigma_floor: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        CEM 参数说明（大白话）：
          - iters：迭代轮数，每轮都会采样 population 组参数；
          - elite_frac：取前 20% 作为精英，更新均值/方差；
          - smoothing：参数平滑（0.8 表示新均值=0.8*旧 + 0.2*精英均值）；
          - sigma_floor：方差地板，防止过早收敛为 0。
        """
        # 初始均值 / 方差
        mean = (init_mean or PolicyParams()).as_vector()
        sigma = list(init_sigma or [0.15, 0.15, 0.15, 0.15, 0.08, 150.0])  # 最后一维是 max_kw_cap 的 std

        best_params = PolicyParams.from_vector(mean)
        best_score = -1e18
        history: List[Dict[str, Any]] = []

        for it in range(iters):
            # 采样族群
            samples: List[Tuple[PolicyParams, float]] = []
            for _ in range(population):
                vec = []
                for m, s in zip(mean, sigma):
                    # 使用高斯采样（random.gauss）
                    vec.append(random.gauss(m, max(s, sigma_floor)))
                cand = PolicyParams.from_vector(vec)
                res = self.evaluate_params(cand, episodes=episodes_per_eval)
                samples.append((cand, res["avg_return"]))

            # 排序 & 精英
            samples.sort(key=lambda x: x[1], reverse=True)
            elites = samples[:max(1, int(elite_frac * population))]

            # 记录本轮
            round_best = elites[0]
            history.append({
                "iter": it,
                "round_best": {"score": round_best[1], "params": asdict(round_best[0])},
                "mean": mean,
                "sigma": sigma,
            })

            # 更新全局最佳
            if round_best[1] > best_score:
                best_score = round_best[1]
                best_params = round_best[0]

            # 计算精英均值/方差
            dim = len(mean)
            elite_vecs = [[(e[0].as_vector())[d] for e in elites] for d in range(dim)]
            elite_means = [sum(v)/len(v) for v in elite_vecs]
            elite_vars = []
            for d in range(dim):
                m = elite_means[d]
                var = sum((x - m) ** 2 for x in elite_vecs[d]) / max(1, len(elite_vecs[d]) - 1)
                elite_vars.append(var)
            elite_sigmas = [math.sqrt(max(v, sigma_floor)) for v in elite_vars]

            # 平滑更新
            mean = [smoothing * m_old + (1 - smoothing) * m_new for m_old, m_new in zip(mean, elite_means)]
            sigma = [max(sigma_floor, smoothing * s_old + (1 - smoothing) * s_new) for s_old, s_new in zip(sigma, elite_sigmas)]

        return {"best_params": asdict(best_params), "best_score": best_score, "history": history, "algo": "cem"}

    # ---- 随机搜索（保留以兼容旧占位） ----
    def random_search(self, iters: int = 40, episodes_per_eval: int = 3, init: Optional[PolicyParams] = None) -> Dict[str, Any]:
        best_params = init or PolicyParams()
        best_score = -1e18
        history: List[Dict[str, Any]] = []
        for i in range(iters):
            cand = PolicyParams(
                bess_charge_scale=max(0.3, min(1.8, best_params.bess_charge_scale * random.uniform(0.85, 1.15))),
                bess_discharge_scale=max(0.3, min(1.8, best_params.bess_discharge_scale * random.uniform(0.85, 1.15))),
                agv_charge_scale=max(0.3, min(1.8, best_params.agv_charge_scale * random.uniform(0.85, 1.15))),
                lighting_reduce_scale=max(0.3, min(1.8, best_params.lighting_reduce_scale * random.uniform(0.85, 1.15))),
                chiller_delta_scale=max(0.5, min(1.5, best_params.chiller_delta_scale * random.uniform(0.9, 1.1))),
                max_kw_cap=max(100.0, min(1500.0, best_params.max_kw_cap + random.uniform(-50, 50))),
            )
            res = self.evaluate_params(cand, episodes=episodes_per_eval)
            history.append({"iter": i, "params": asdict(cand), "avg_return": res["avg_return"]})
            if res["avg_return"] > best_score:
                best_score = res["avg_return"]
                best_params = cand
        return {"best_params": asdict(best_params), "best_score": best_score, "history": history, "algo": "random"}

    # ---- CQL/BCQ 占位：默认委托 CEM（或随机搜索），保留兼容接口 ----
    def train_cql(self, dataset_dir: Optional[str], iters: int = 25, episodes_per_eval: int = 3) -> Dict[str, Any]:
        # A full CQL implementation must fit Q and policy models from dataset_dir before evaluation.
        return self.train_cem(iters=iters, episodes_per_eval=episodes_per_eval)

    def train_bcq(self, dataset_dir: Optional[str], iters: int = 25, episodes_per_eval: int = 3) -> Dict[str, Any]:
        return self.random_search(iters=max(30, iters), episodes_per_eval=episodes_per_eval)

    # ---- 保存产物（策略参数 + 元数据） ----
    def save_policy(self, out_dir: str, algo: str, search_result: Dict[str, Any]) -> str:
        ts = int(time.time())
        fname = f"policy-{algo}-{ts}.json"
        fpath = str(Path(out_dir) / fname)
        bundle = {
            "algo": algo,
            "created_at": _now_iso(),
            "best_params": search_result.get("best_params"),
            "best_score": search_result.get("best_score"),
            "history_sample": search_result.get("history", [])[:10],  # 控制产物大小
            "env": "PortEnergyEnvPro",
            "note": "策略参数文件；rl.py 在运行时会加载最新的 policy-*.json 并应用到动作上。"
        }
        _write_json(fpath, bundle)
        return fpath

    # ---- 发布：登记候选 + 进入影子 ----
    def publish_to_rollout(self, policy_path: str, version: Optional[str] = None, auto_shadow: bool = True) -> Dict[str, Any]:
        if RLPolicyRollout is None:
            return {"error": "rollout module not available", "policy_path": policy_path}
        ro = RLPolicyRollout()
        ver = version or Path(policy_path).stem  # 默认用文件名当版本号
        status1 = ro.register_candidate(ver, meta={"policy_path": policy_path})
        status2 = ro.start_shadow() if auto_shadow else {"note": "shadow not started"}
        return {"register": status1, "shadow": status2}


# --------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Offline RL Trainer for PortEnergyEnvPro (CEM / Random / CQL-BCQ placeholder)")
    parser.add_argument("--algo", choices=["cem", "cql", "bcq", "random"], default="cem", help="训练算法：cem（推荐）/ cql / bcq / random")
    parser.add_argument("--dataset", type=str, default="", help="数据集目录（为空则自动用 env_pro 合成）")
    parser.add_argument("--episodes", type=int, default=4, help="合成数据集的 episode 数（仅当 --dataset 为空时生效）")
    parser.add_argument("--iters", type=int, default=25, help="优化迭代数（cem/random）")
    parser.add_argument("--eval-episodes", type=int, default=3, help="每次评估的 episode 数")
    parser.add_argument("--population", type=int, default=32, help="CEM 族群大小")
    parser.add_argument("--elite-frac", type=float, default=0.2, help="CEM 精英比例（0~1）")
    parser.add_argument("--out", type=str, default="data/objects/rl/policies", help="策略产物输出目录")
    parser.add_argument("--publish", action="store_true", help="训练完成后是否自动登记候选并启动影子模式")
    args = parser.parse_args()

    if PortEnergyEnvPro is None:
        raise RuntimeError("缺少 rl_env_pro.PortEnergyEnvPro，请检查 app/services/rl_env_pro.py")

    # 1) 数据集：若未提供，自动用 Pro 环境合成（合成数据主要用于留存与追责，不直接用于 CEM）
    dataset_dir = args.dataset
    if not dataset_dir:
        env = PortEnergyEnvPro()
        ds_dir = "data/objects/rl/datasets"
        env.rollouts_pro(episodes=max(1, args.episodes), out_dir=ds_dir)
        dataset_dir = ds_dir
        print(f"[INFO] 合成数据集保存在：{dataset_dir}")

    # 2) 训练
    trainer = OfflineRLTrainer()
    if args.algo == "cem":
        result = trainer.train_cem(
            iters=args.iters,
            population=args.population,
            elite_frac=args.elite_frac,
            episodes_per_eval=args.eval_episodes,
        )
    elif args.algo == "cql":
        result = trainer.train_cql(dataset_dir=dataset_dir, iters=args.iters, episodes_per_eval=args.eval_episodes)
    elif args.algo == "bcq":
        result = trainer.train_bcq(dataset_dir=dataset_dir, iters=args.iters, episodes_per_eval=args.eval_episodes)
    else:
        result = trainer.random_search(iters=args.iters, episodes_per_eval=args.eval_episodes)

    # 3) 产物落盘
    policy_path = trainer.save_policy(args.out, algo=result.get("algo", args.algo), search_result=result)
    print(f"[INFO] 策略产物：{policy_path}")
    print(f"[INFO] 最佳得分：{result.get('best_score')}")

    # 4) （可选）自动登记候选并进入影子
    if args.publish:
        pub = trainer.publish_to_rollout(policy_path, version=None, auto_shadow=True)
        print("[INFO] 已登记候选并启动影子：")
        print(json.dumps(pub, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
