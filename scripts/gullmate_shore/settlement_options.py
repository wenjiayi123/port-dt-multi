"""Daily control intervals cross the tariff boundary and prohibit fictitious FIFO repayments."""
from datetime import datetime
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from options import dispatch,forecast

MENU=(
 ("idle",0.,0.,0.,0.),
 ("flex_bill",0.,.98,.000014,.7),
 ("flex_strict",0.,.98,.000024,.85),
 ("flex_peak",0.,.98,.000016,1.3),
 ("joint_small",280.,.98,.000016,.8),
 ("joint_large",650.,.98,.000020,1.3),
 ("joint_guard",400.,.94,.000030,.90),
 ("storage_only",650.,0.,0.,1.0),
)


def peak_profile(env):
    x=env.segment[:,0].astype(float).reshape(-1,24)
    gap=x.max(axis=1)[:,None]-x
    return {"utc_gap_kw":np.quantile(gap,.025,axis=0).tolist(),"quantile":.025,"training_days":len(x),
            "note":"Conservative train-only distance to daily peak, not a guaranteed confidence limit."}


@lru_cache(maxsize=3)
def matrix(h):
    src,dst=map(np.asarray,zip(*[(i,j) for i in range(h) for j in range(i+1,min(h,i+13))]))
    n=2*h+len(src)+2;t=np.arange(h);arc=2*h+np.arange(len(src))
    g=np.zeros((h,n));g[t,t]=1;g[t,h+t]=-1;g[src,arc]=-1;g[dst,arc]=1
    defer=np.zeros_like(g);repay=np.zeros_like(g);defer[src,arc]=1;repay[dst,arc]=1
    return src,dst,g,defer,repay


def plan(env,clock,choice,h):
    _,power,share,cut,peak_weight=MENU[choice]
    if choice==0:return np.zeros((h,2)),{"carbon_reserve_short_kg":0.,"planned_peak_kw":0.}
    src,dst,g,defer,repay=matrix(h);n=g.shape[1]
    load,price,carbon=(v[:h] for v in forecast(env,clock))
    stamp=datetime.fromisoformat(env.timestamps[env._start+env._step].replace("Z","+00:00"))
    hours=(stamp.hour+np.arange(h))%24
    gap=np.maximum(0.,np.asarray(clock["peak_profile"]["utc_gap_kw"])[hours]-40.)
    peak=g.copy();peak[:,-2]=1.
    cap=share*env.defer_limit_kw;upper=np.full(n,np.inf);upper[:2*h]=power;upper[2*h:-2]=cap
    # Exclude signal accumulation at the terminal episode boundary.
    upper[2*h:-2][env._step+src>=env.episode_steps-12]=0.
    upper[-2]=min(350.,cap+power)
    inv=np.zeros_like(g);inv[:,:h]=np.tri(h)*env.charge_eff;inv[:,h:2*h]=-np.tri(h)/env.discharge_eff
    remaining=env.episode_steps-env._step-np.arange(1,h+1)
    band=np.minimum(.18*remaining/(env.episode_steps-1),remaining*env.power_kw*min(env.charge_eff,env.discharge_eff)/env.energy_kwh)
    up=(np.minimum(env.soc_max,env.soc_initial+band)-env._soc)*env.energy_kwh
    down=(env._soc-np.maximum(env.soc_min,env.soc_initial-band))*env.energy_kwh
    co2=carbon@g
    target=co2.copy();target[-1]=-1.
    # Hard non-regression with a small numerical margin, plus a separately
    # reported shortage if the preferred positive carbon saving is unavailable.
    margin=co2.copy();margin[:-2]+=.0001
    A=csr_matrix(np.vstack([peak,g,defer,repay,inv,-inv,target,margin]))
    rhs=np.r_[gap,env.hard_pcc_limit_kw-load,np.full(2*h,cap),up,down,-cut*(load@carbon),0.]
    obj=price@g;obj[:2*h]+=env.cycle_cost
    rate=env.config["grid"]["demand_charge_cny_per_kw_month"]*env.episode_steps/(24*30.4375)
    obj[-2]=-rate*peak_weight/7.;obj[-1]=60.
    kw=dict(A_ub=A,b_ub=rhs,A_eq=csr_matrix(inv[-1:]),b_eq=[(env.soc_initial-env._soc)*env.energy_kwh],method="highs")
    def solve():
        solution=linprog(obj,bounds=np.column_stack([np.zeros(n),upper]),**kw)
        if not solution.success:raise ValueError("Clock-aligned option failed: "+solution.message)
        return solution.x
    relaxed=solve()
    net=(repay-defer)@relaxed
    # A single native auxiliary power field cannot reset an old task's
    # deadline by simultaneously deferring and repaying at the same instant.
    upper[2*h:-2][(net[src]>=-1e-7)|(net[dst]<=1e-7)]=0.
    x=solve()
    flex=(repay-defer)@x;discharge=x[h:2*h]-x[:h]
    assert np.max(np.minimum(repay@x,defer@x))<1e-6
    return np.column_stack([discharge,flex]),{"carbon_reserve_short_kg":float(x[-1]),"planned_peak_kw":float(x[-2]),
              "clock_start_utc":stamp.hour,"duration":h,"planned_carbon_delta_kg":float(co2@x)}


def run(env,clock,choice,trace=None):
    # Seven model decisions per 168 hours: 31 hours from the first choice,
    # five aligned 24-hour periods, and a final 17-hour period. The first
    # choice re-plans at hour seven using then-current measurements.
    parts=(7,24) if env._step==0 else (min(24,env.episode_steps-env._step),)
    money=carbon=reward=0.;segments=[]
    for n in parts:
        scheduled,ss=plan(env,clock,choice,n);segments.append(ss)
        for h,(bess,flex) in enumerate(scheduled):
            c=env._row_context();soc=env._soc
            if h==n-1:
                gap=(env._soc-env.soc_initial)*env.energy_kwh
                bess=gap*env.discharge_eff if gap>0 else gap/env.charge_eff
                flex=env._flex_backlog_kwh
            action=dispatch(env,c,bess,flex)
            _,r,end,trunc,i=env.step(action);assert not trunc and end==(env._step==env.episode_steps)
            money+=i["v8_reward"]["bill_delta_cny"];carbon+=i["v8_reward"]["carbon_delta_kg"];reward+=r
            if trace is not None:
                trace.append({"hour":env._step-1,"option":MENU[choice][0],"bess_kw":i["final_action"]["bess_kw"],"flex_kw":i["final_action"]["flex_kw"],
                    "base_kw":c["base_load_kw"],"pcc_kw":i["pcc_kw"],"soc_before":soc,"soc":i["soc"],"price":c["price_cny_per_kwh"],
                    "factor":c["carbon_kg_per_kwh"],"mandatory_shore_kw":c["mandatory_shore_kw"],"auxiliary_kw":c["auxiliary_shore_kw"],
                    "service_short_kw":i["shore_sla_shortfall_kw"],"reserve_short_kw":i["reserve_shortfall_kw"],"backlog":env._flex_backlog_kwh,
                    "overdue_kwh":i["flex_overdue_kwh"],"guardrail":i["guardrail_violation"],"physical_violation":i["physical_power_violation"],
                    "latent":action.tolist(),"requested_kw":[float(bess),float(flex)]})
    return {"cost_delta":money,"carbon_delta":carbon,"native_reward":reward,"day_end_soc":env._soc,"day_end_backlog":env._flex_backlog_kwh,
            "native_hours":sum(parts),"planning_intervals":segments}
