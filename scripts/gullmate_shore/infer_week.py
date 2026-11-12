"""Recompute a registered or new UTC week with the frozen actor in the native TwinLab library."""
from datetime import date
import json
from pathlib import Path
import sys
import torch
from native import PERIODS,history,plant,read,sha
from options import observation
from settlement_options import run,MENU

HERE=Path(__file__).resolve().parent
MODEL_SHA="642b5a6f56c5d07ad8e17bfda415c73dfecd02c6498e2d21aaf6dcd443eadcb1"
MANIFEST_SHA="5f40e3b2136b8fa20ef3f2e11776436ddd03d99bf3795d5113380ca1d596a4e2"


def simulate(day):
    stamp=date.fromisoformat(day)
    part=next((k for k,(a,b) in PERIODS.items() if k!="train" and a<=day<b),None)
    if not part:raise ValueError("日期不在支持的验证、重复测试或前向分区")
    if sha(HERE/"dispatch.ts")!=MODEL_SHA or sha(HERE/"manifest.json")!=MANIFEST_SHA:raise ValueError("岸电策略或训练参数哈希不符")
    meta=read(HERE/"manifest.json")
    if history()[1].fingerprint!=meta["history_sha256"]:raise ValueError("历史环境数据指纹不符")
    env=plant(part)
    if env.dataset.fingerprint!=meta["dataset_sha"][part]:raise ValueError("当前仿真分区数据指纹不符")
    model=torch.jit.load(str(HERE/"dispatch.ts"),map_location="cpu").eval();clock=meta["clock"]
    try:
        try:start=next(i for i,t in enumerate(env.timestamps) if t[:10]==day and t[11:13]=="00")
        except StopIteration as e:raise ValueError("未找到该 UTC 周起点") from e
        if start+168>=len(env.segment):raise ValueError("周窗口越过冻结分区，缺少完整的 168 小时及后继观测")
        results={}
        with torch.inference_mode():
            for method in ("idle","selected","best_fixed"):
                _,rst=env.reset(options={"start_index":start});assert rst["start_index"]==start
                series=[];days=[]
                for d in range(7):
                    choose=int(model(torch.from_numpy(observation(env,clock))).argmax()) if method=="selected" else meta["best_fixed"] if method=="best_fixed" else 0
                    result=run(env,clock,choose,series)
                    days.append({"decision":d+1,"option":MENU[choose][0],"duration_hours":result["native_hours"]})
                totals=env.totals
                metrics={k:float(totals[k]) for k in ("total_cost_cny","carbon_kg","peak_kw","terminal_soc_error","terminal_flex_backlog_kwh","bess_throughput_kwh","guardrail_violations")}
                results[method]={"metrics":metrics,"options":days,"hours":series}
        return {"week":day,"part":part,"hours":168,"weight_sha256":meta["weight"],"runtime_sha256":MODEL_SHA,
                "methods":results,"production_authority":False,"scope":"原生 TwinLab 离线仿真，不操作设备。"}
    finally:env.close()


if __name__=="__main__":
    try:print(json.dumps(simulate(sys.argv[1]),ensure_ascii=False,allow_nan=False))
    except (OSError,ValueError) as e:sys.exit("岸电试算失败："+str(e))
