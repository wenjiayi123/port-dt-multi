from datetime import datetime
import torch
import numpy as np
from options import forecast,observation,dispatch
from settlement_options import plan as original
TOL=.005
MIN_OBSERVATIONS=6
COLUMNS=('base_kw', 'price', 'carbon_factor', 'pcc_kw', 'soc', 'battery_kw', 'flex_kw', 'backlog_kwh', 'service_short_kw', 'reserve_short_kw', 'overdue_kwh', 'guardrail', 'physical')
METRICS=('total_cost_cny', 'carbon_kg', 'peak_kw', 'terminal_soc_error', 'terminal_flex_backlog_kwh', 'guardrail_violations')

def make_clock(env,clock,stamp,day,arm,span):
    from options import forecast
    if arm=="original":return clock
    load,_,_=forecast(env,clock);values=np.array(clock["profiles"])
    gaps=np.array(clock["peak_profile"]["utc_gap_kw"])
    hours=(stamp.hour+np.arange(24))%24
    for k,h in enumerate(hours):values[h,0]=load[k]
    usable=min(24-stamp.hour,span)
    known=day["hours"][stamp.hour:stamp.hour+usable]
    assert known[0]["utc"][:13]==stamp.isoformat()[:13] and day["day"]==stamp.date().isoformat()
    for k,row in enumerate(known):
        if k:values[hours[k],0]=row["p50"]
    if arm in ("point_headroom","interval_headroom"):
        # Use only the day already issued. A lower bound on its maximum and upper
        # bound for a given hour produce a conservative headroom estimate.
        low,high=("p10","p90") if arm=="interval_headroom" else ("p50","p50")
        peak=max(max(v[low] for v in day["hours"]),load[0])
        for k,row in enumerate(known):
            upper=row[high] if k else load[0]
            gaps[hours[k]]=max(0.,peak-upper)
    return {**clock,"profiles":values.tolist(),"peak_profile":{**clock["peak_profile"],"utc_gap_kw":gaps.tolist()}}

def identify(clock,paid,hour):
    pattern=np.asarray(clock,dtype=float);readings=np.asarray(paid[-24:],dtype=float)
    if pattern.shape!=(24,) or not len(readings) or not np.isfinite(readings).all():
        return {"phase":None,"status":"missing","matched":0,"observations":len(readings)}
    hours=(hour+1-len(readings)+np.arange(len(readings)))%24
    candidates=[]
    for shift in range(24):
        error=np.abs(readings-pattern[(hours-shift)%24])
        mismatches=np.flatnonzero(error>TOL)
        tail=len(error) if not len(mismatches) else len(error)-int(mismatches[-1])-1
        candidates.append(tail)
    best=max(candidates);winners=np.flatnonzero(np.array(candidates)==best)
    if best>=MIN_OBSERVATIONS and len(winners)==1 and np.ptp(readings[-best:])>2*TOL:
        return {"phase":int(winners[0]),"status":"confirmed","matched":best,"observations":len(readings)}
    # Start in native mode only while all observations received so far fit the native tariff.
    unchanged=candidates[0]==len(readings)
    return {"phase":0 if unchanged else None,"status":"native_consistent" if unchanged else "unconfirmed",
            "matched":best,"observations":len(readings)}

def schedule(env,clock,ts,day,arm,span,choice,bounded):
    ref,_=original(env,clock,choice,span)
    info={"choice":choice,"at":ts.isoformat(),"span":span,"fallback":False}
    if arm in ("idle","original") or choice==0:return ref,info
    proposed=make_clock(env,clock,ts,day,"point_headroom",span)
    net=ref[:,1]-ref[:,0];carbon=forecast(env,clock)[2][:span];ceiling=float(net@carbon)-.1
    cap=None
    if arm=="peak_guard":
        cap=net.copy()
        covered=min(span,24-ts.hour);seen=float(env.totals.get("peak_kw",0.))
        upper=np.array([r["p90"] for r in day["hours"][ts.hour:ts.hour+covered]])
        upper[0]=env._row_context()["base_load_kw"]
        cap[:covered]=np.maximum(net[:covered],seen-upper)
        info.update(available_peak_kw=seen,known_day_hours=covered)
    try:actions,_=bounded(env,proposed,choice,span,ceiling,cap)
    except ValueError as e:
        if "infeasible" not in str(e).lower():raise
        info["fallback"]=True;return ref,info
    actual_net=actions[:,1]-actions[:,0]
    assert float(actual_net@carbon)<=ceiling+1e-5
    if cap is not None:assert np.max(actual_net-cap)<1e-5
    return actions,info

def episode(env,start,arm,days,actor,clock,bounded,decision=schedule):
    _,state=env.reset(options={"start_index":start});assert state["start_index"]==start
    hourly=[];plans=[]
    for j in range(7):
        with torch.inference_mode():choice=0 if arm=="idle" else int(actor(torch.from_numpy(observation(env,clock))).argmax())
        for span in ((7,24) if j==0 else (min(24,168-env._step),)):
            ts=datetime.fromisoformat(env.timestamps[start+env._step].replace("Z","+00:00"))
            act,info=decision(env,clock,ts,days[ts.date().isoformat()],arm,span,choice,bounded);plans.append(info)
            for i,(bess,flex) in enumerate(act):
                ctx=env._row_context()
                if i==span-1:
                    gap=(env._soc-env.soc_initial)*env.energy_kwh
                    bess=gap*env.discharge_eff if gap>0 else gap/env.charge_eff
                    flex=env._flex_backlog_kwh
                _,_,ended,truncated,v=env.step(dispatch(env,ctx,bess,flex));assert not truncated and ended==(env._step==168)
                hourly.append([ctx["base_load_kw"],ctx["price_cny_per_kwh"],ctx["carbon_kg_per_kwh"],v["pcc_kw"],v["soc"],v["final_action"]["bess_kw"],
                               v["final_action"]["flex_kw"],env._flex_backlog_kwh,v["shore_sla_shortfall_kw"],v["reserve_shortfall_kw"],v["flex_overdue_kwh"],
                               bool(v["guardrail_violation"]),bool(v["physical_power_violation"])])
    return {"columns":list(COLUMNS),"hourly":hourly,"plans":plans,"metrics":{k:float(env.totals[k]) for k in METRICS}}

def decide(env,clock,ts,day,arm,span,choice,bounded):
    last=env._start+env._step
    rates=np.asarray(clock["profiles"])[:,1]
    history=env.segment[max(0,last-23):last+1,4]
    result=identify(rates,history,ts.hour)
    phase=result["phase"] if arm=="adaptive_clock" else 0
    selected=0 if phase is None else choice
    use=clock
    if phase:
        modified=np.array(clock["profiles"]);modified[:,1]=np.roll(rates,phase)
        use={**clock,"profiles":modified.tolist()}
    actions,_=original(env,use,selected,span)
    return actions,{"at":ts.isoformat(),"choice":choice,"applied_choice":selected,"span":span,"fallback":phase is None,
                    "observed":result,"phase_used":phase,"current_price":float(history[-1]),"expected_native_price":float(rates[ts.hour])}
