"""Run a new 168-hour comparison using the managed TwinLab plant (no site control)."""
import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
TWIN=HERE.parents[1]
DATA=HERE/"data"
NATIVE=TWIN/"scripts/gullmate_shore"
CASES=("shanghai","industrial_mix","industrial_mix_price_shift","phase_minus3","phase_plus6","week_switch","unlisted_tariff")


def checked(p,code):
    b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=code:raise ValueError("数据或模型指纹不符："+p.name)
    return b


def run(case,day):
    if case not in CASES:raise ValueError("未登记的负荷或电价工况")
    manifest=json.loads(checked(DATA/"index.json","9cbe54a8f267d775f350c33970ce6002cfb2e2b8f76e799596fe027ac569739c"))
    for path,sha in manifest["data"].items():checked(DATA/path,sha)
    for path,sha in manifest["native"].items():checked(NATIVE/path,sha)
    for path,sha in manifest["code"].items():checked(HERE/path,sha)
    sys.path.insert(0,str(NATIVE));sys.path.insert(0,str(TWIN))
    from native import history,train
    from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
    from algorithms import episode,schedule,decide
    from bounded_external import plan
    cfg,hist,scale=history();prior=train.load_port_dataset("public_cn_sha_forward_2026m05_v1",data_root=TWIN/"data/rl/datasets")
    if [hist.fingerprint,prior.fingerprint]!=manifest["historical_data"]:raise ValueError("原生训练或历史基准指纹不符")
    combined=train.joined_forward(hist,prior);d=train.load_port_dataset("shanghai_2026_jun_sep_external",data_root=DATA)
    arr=d.values.copy()
    if case!="shanghai":arr[:,0]=json.loads((DATA/"mix-kw.json").read_bytes())
    idx=np.arange(len(arr));prices=arr[:,4].reshape(-1,24)
    shift={"industrial_mix_price_shift":4,"phase_minus3":-3,"phase_plus6":6}.get(case,0)
    if case=="week_switch":shift=np.where((idx%168>=60)&(idx%168<132),-3,0)
    if case=="unlisted_tariff":arr[:,4]=arr[:,4]*1.18
    else:arr[:,4]=prices[idx//24,(idx-shift)%24]
    start=(day-date(2026,6,1)).days*24
    if start<0 or start+168>=len(arr):raise ValueError("该日期不能形成完整研究周及后继状态")
    env=ShoreBESSV8SACEnv(train.joined_forward(combined,replace(d,values=arr)),slice(len(combined.timestamps),None),config=cfg,
                           normalization_slice=scale,episode_steps=168,carbon_price=16.,training=False,seed=0,record_trace=False)
    days=json.loads((DATA/("shanghai-days.json" if case=="shanghai" else "mix-days.json")).read_bytes())
    actor=torch.jit.load(str(NATIVE/"dispatch.ts"),map_location="cpu").eval()
    model=json.loads((NATIVE/"manifest.json").read_bytes())
    methods=["idle","original","adaptive_clock"]
    if case in CASES[:3]:methods.extend(("paired_cap_margin","peak_guard"))
    try:
        records={m:episode(env,start,m,days,actor,model["clock"],plan,decision=decide if m=="adaptive_clock" else schedule) for m in methods}
    finally:env.close()
    return {"schema":"gullmate.external-week.v1","case":case,"week":day.isoformat(),"weight":model["weight"],"source_sha":d.fingerprint,
            "production_authority":False,"records":records}


if __name__=="__main__":
    try:print(json.dumps(run(sys.argv[1],date.fromisoformat(sys.argv[2])),ensure_ascii=False,separators=(",",":"),allow_nan=False))
    except (ValueError,OSError) as e:sys.exit("工程复算未完成："+str(e))
