"""Midnight 24-hour forecasts, from hourly observations available before issuance."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np

HOUR=3600
SITES={"shanghai":"Asia/Shanghai","los_angeles":"America/Los_Angeles","funchal":"Atlantic/Madeira"}
BASELINES=("last","recent_day","recent_week","profile","native_ar")


def inputs(history, origin, site):
    # Index zero is the target origin; its preceding UTC hour is embargoed.
    values=np.asarray([history.get(origin-k*HOUR,np.nan) for k in range(169,1,-1)],dtype=float)
    if not np.isfinite(values[-24:]).all():raise ValueError("签发时刻前缺少连续 24 个已完成的小时读数")
    if np.any(values[np.isfinite(values)]<0):raise ValueError("负有功功率不适用当前模型")
    scale=max(float(np.nanmedian(values)),1e-3)
    def past(offset):
        # offset is relative to the target origin, and must precede the embargo.
        assert -169<=offset<=-2
        return values[offset+169]
    predictions={k:[] for k in BASELINES[:-1]};features=[]
    for h in range(24):
        seasonal=[past(h-24*k) if h-24*k<=-2 else np.nan for k in range(1,8)]
        available=[v for v in seasonal if np.isfinite(v)]
        recent=available[0] if available else values[-1]
        weekly=seasonal[-1] if np.isfinite(seasonal[-1]) else recent
        profile=float(np.median(available)) if available else recent
        for key,x in zip(predictions,(values[-1],recent,weekly,profile),strict=True):predictions[key].append(float(x))
        day=datetime.fromtimestamp(origin+h*HOUR,timezone.utc).astimezone(ZoneInfo(SITES[site]))
        phase=2*np.pi*day.hour/24
        features.append([recent/scale-1,weekly/scale-1,profile/scale-1,
                         values[-1]/scale-1,values[-6:].mean()/scale-1,values[-24:].mean()/scale-1,
                         np.nanmean(values)/scale-1,len(available)/7, float(day.weekday()>=5),
                         np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase),h/23])
    return np.asarray(features),scale,{k:np.asarray(v) for k,v in predictions.items()},values


def train(examples,penalty):
    X=np.concatenate([e["x"] for e in examples])
    y=np.concatenate([(e["actual"]-e["base"]["profile"])/e["scale"] for e in examples])
    center=X.mean(axis=0);offset=float(y.mean());xc=X-center
    w=np.linalg.solve(xc.T@xc+float(penalty)*np.eye(X.shape[1]),xc.T@(y-offset))
    return {"weights":w.tolist(),"center":center.tolist(),"bias":offset,"penalty":penalty}


def predict(x,scale,base,model):
    residual=(x-np.asarray(model["center"]))@np.asarray(model["weights"])+model["bias"]
    return np.maximum(0,base["profile"]+scale*residual)


def interval(pred,scale,ranges):
    q=np.asarray(ranges).repeat(6,axis=0)
    return np.maximum(0,pred+scale*q[:,0]),np.maximum(0,pred+scale*q[:,1])
