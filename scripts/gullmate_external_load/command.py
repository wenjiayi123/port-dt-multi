"""Operator entry point for a native hourly policy comparison and optional network readback."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen
from native_worker import run,CASES


def main(args):
    p=argparse.ArgumentParser(description="数字孪生 · 新月份与外部负荷压力工况")
    p.add_argument("case",choices=CASES);p.add_argument("week",type=date.fromisoformat);p.add_argument("--verify-twin",action="store_true");p.add_argument("--json",action="store_true")
    a=p.parse_args(args);r=run(a.case,a.week)
    if a.verify_twin:
        url=f"http://127.0.0.1:8000/api/forecast/external-load/{a.case}/{a.week.isoformat()}"
        try:
            with urlopen(url,timeout=85) as response:
                if response.geturl()!=url:raise ValueError("孪生地址不符")
                b=response.read(1500001)
        except (URLError,OSError) as e:raise ValueError("TwinLab 未就绪或无法读回") from e
        if len(b)>1500000 or json.loads(b)!=r:raise ValueError("逐小时或结算结果不一致")
        r["verified_twin"]=True
    if a.json:print(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(",",":")));return
    print(a.week,a.case,"· 168 小时上海设备工程复算 · 现场权限关闭")
    for name,v in r["records"].items():
        m=v["metrics"];print(f"{name:22s} {m['total_cost_cny']:,.2f} 元  {m['carbon_kg']:,.3f} kg  {m['peak_kw']:,.2f} kW")
    if a.verify_twin:print("已与本次 TwinLab 独立 HTTP 重算逐值核对。")


if __name__=="__main__":
    try:main(sys.argv[2:] if sys.argv[1:2]==["stress"] else sys.argv[1:])
    except (ValueError,OSError) as e:sys.exit(str(e))
