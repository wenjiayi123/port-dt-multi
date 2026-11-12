"""Local operator command; compare exact native receipts only if explicitly requested."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen

TWIN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TWIN))
from app.services.port_load_outlook.summer_forecast import forecast


def main(args):
    p=argparse.ArgumentParser(description="港口日前负荷 · 原生工程与公开电表回放")
    p.add_argument("site",choices=("shanghai","los_angeles","funchal"))
    p.add_argument("day",type=date.fromisoformat)
    p.add_argument("--verify-twin",action="store_true")
    p.add_argument("--daily-peak",action="store_true",help="单独估计日峰值，不指定出现时刻")
    p.add_argument("--json",action="store_true")
    a=p.parse_args(args);result=forecast(a.site,a.day,assess_peak=a.daily_peak)
    if a.verify_twin:
        url=f"http://127.0.0.1:8000/api/forecast/port-load/{a.site}/{a.day.isoformat()}"
        if a.daily_peak:url+="?assess_peak=true"
        try:
            with urlopen(url,timeout=15) as resp:
                if resp.geturl()!=url:raise ValueError("孪生回执地址不符")
                raw=resp.read(100001)
        except (OSError,URLError) as exc:raise ValueError("官方 TwinLab 当前不能读回该日期") from exc
        if len(raw)>100000 or json.loads(raw)!=result:raise ValueError("官方 TwinLab 回执与本机原生重算不符")
        result["verified_twin"]={"endpoint":url,"hours":24,"numeric_forecast_fields":72}
    if a.json:print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    else:
        print(f"{a.site}  {a.day} UTC  ·  {result['source_kind']}  ·  {result['point_method']}")
        print("UTC 小时起点                 下界 kW     中心 kW     上界 kW")
        for h in result["hours"]:print(f"{h['utc']} {h['p10']:11.2f} {h['p50']:11.2f} {h['p90']:11.2f}")
        print(f"日峰值预测 {result['peak_kw']:.2f} kW · 反馈 {len(result['feedback_dates'])} 天 · 现场权限关闭")
        if a.daily_peak:print(f"独立日峰值估计 {result['peak_assessment']['daily_peak_kw']:.2f} kW · 未估计峰值时刻")
        if a.verify_twin:print("本次已通过 TwinLab 独立核对。")


if __name__=="__main__":
    try:main(sys.argv[2:] if sys.argv[1:2]==["forecast"] else sys.argv[1:])
    except (OSError,ValueError) as exc:sys.exit(str(exc))
