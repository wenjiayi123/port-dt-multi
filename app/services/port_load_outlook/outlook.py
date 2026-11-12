"""Local day-ahead forecast. Only completed historical observations enter fitted predictions or calibration."""
import csv
import hashlib
import json
from datetime import datetime,time,timezone
from io import StringIO
from pathlib import Path
import numpy as np
from .forecasting import SITES,inputs,predict

DAY=86400
HOUR=3600
PIN="6fcb5bb561f4bba5bf93b98f73523fab4b264d186a469f6884ada1b032e57323"
DATA=Path(__file__).resolve().parent/"data"


def verified(path,wanted):
    b=path.read_bytes()
    if hashlib.sha256(b).hexdigest()!=wanted:raise ValueError("预测档案指纹不符")
    return b


def prepared(site,root=DATA):
    if site not in SITES:raise ValueError("未登记的港区")
    manifest=json.loads(verified(root/"manifest.json",PIN))
    spec=json.loads(verified(root/"profiles.json",manifest["profiles.json"]))[site]
    key=site+".csv";b=verified(root/key,manifest[key])
    rows={int(datetime.fromisoformat(v["utc_start"]).timestamp()):float(v["mean_kw"]) for v in csv.DictReader(StringIO(b.decode()))}
    return spec,rows,manifest[key]


def point(data,origin,site,profile):
    x,scale,base,_=inputs(data,origin,site)
    method=profile["point_method"]
    pred=predict(x,scale,base,profile["model"]) if method=="local" else base[method]
    return pred,scale


def issue(site,day,spec,history):
    origin=int(datetime.combine(day,time.min,timezone.utc).timestamp())
    if day.isoformat()<spec["valid_from"]:raise ValueError("该日早于训练、选择和校准截止日")
    # The oldest eligible feedback day also needs its own complete 169-hour lookback.
    hist={t:history[t] for t in range(origin-37*DAY,origin-HOUR,HOUR) if t in history}
    pred,scale=point(hist,origin,site,spec)
    errors=[];origins=[]
    for offset in range(29,1,-1):
        prior=origin-offset*DAY
        if prior<spec["feedback_from"]:continue
        try:
            old,s=point(hist,prior,site,spec)
            observed=np.array([hist[prior+h*HOUR] for h in range(24)])
        except (KeyError,ValueError):continue
        errors.append((observed-old)/s)
        origins.append(datetime.fromtimestamp(prior,timezone.utc).date().isoformat())
    if len(errors)<4:raise ValueError("此前 28 个可反馈日内缺少至少四个完整预测及实绩日")
    e=np.stack(errors)
    q=np.array([np.quantile(e[:,i:i+6],[.1,.9]) for i in range(0,24,6)]).repeat(6,axis=0)
    low=np.maximum(0,pred+scale*np.minimum(0,q[:,0]))
    high=np.maximum(0,pred+scale*np.maximum(0,q[:,1]))
    fmt=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat().replace("+00:00","Z")
    return {"site":site,"day":day.isoformat(),"source_kind":spec["source_kind"],"point_method":spec["point_method"],
            "issued_at":fmt(origin),"last_input_hour_start":fmt(max(hist)),"feedback_dates":origins,
            "weight_sha256":spec["model_sha256"] if spec["point_method"]=="local" else None,
            "production_authority":False,"hours":[{"utc":fmt(origin+h*HOUR),"p10":float(low[h]),"p50":float(pred[h]),
            "p90":float(high[h])} for h in range(24)],"peak_kw":float(max(pred))}


def forecast(site,day):
    spec,history,source=prepared(site)
    return {**issue(site,day,spec,history),"source_sha256":source}
