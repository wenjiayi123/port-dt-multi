"""Return a separately calibrated daily peak on request, alongside the 24 hourly centers."""
import csv
from datetime import datetime,timezone
from io import StringIO
from statistics import median

from .outlook import DATA,issue,prepared,point,verified

EXTRA_SHA="e3a0e72ed7343103f253acbf2d7eae87527f39b0c56b8884038f07a254b3dc98"


def forecast(site,day,*,assess_peak=False):
    spec,history,source=prepared(site)
    sources={"source_sha256":source}
    if site=="shanghai" and day.isoformat()>"2026-06-01":
        raw=verified(DATA/"shanghai-summer-2026.csv",EXTRA_SHA).decode()
        extension={int(datetime.fromisoformat(v["utc_start"]).timestamp()):float(v["mean_kw"]) for v in csv.DictReader(StringIO(raw))}
        if extension.keys() & history.keys():raise ValueError("新旧预测数据时间重叠")
        history.update(extension);sources["extension_sha256"]=EXTRA_SHA
    r={**issue(site,day,spec,history),**sources}
    if assess_peak:
        residuals=[]
        for previous in r["feedback_dates"]:
            origin=int(datetime.fromisoformat(previous).replace(tzinfo=timezone.utc).timestamp())
            pred,_=point(history,origin,site,spec)
            high=max(history[origin+h*3600] for h in range(24))
            residuals.append((high-float(pred.max()))/max(1e-6,float(pred.mean())))
        centers=[v["p50"] for v in r["hours"]]
        r["peak_assessment"]={"method":"delayed_normalized_daily_peak_residual","daily_peak_kw":r["peak_kw"]+
                              max(0.,median(residuals))*sum(centers)/len(centers),"curve_max_kw":r["peak_kw"],
                              "feedback_days":len(residuals),"last_feedback_day":r["feedback_dates"][-1],"peak_time":None}
    return r
