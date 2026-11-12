"""Day-level dispatch options act through the unchanged native V8 safety projection."""
from datetime import datetime
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

# name, maximum storage kW, flexible service share, minimum forecast carbon
# saving per kWh shifted, carbon shadow weight, marginal demand-charge weight.
OPTIONS=(
    ("idle",0.,0.,0.,0.,0.),
    ("flex_bill",0.,.90,.001,0.,.6),
    ("flex_carbon",0.,.90,.004,12.,.6),
    ("flex_peak",0.,.90,.001,3.,1.),
    ("joint_300",300.,.90,.001,2.,.65),
    ("joint_700",700.,.90,.001,2.,.65),
    ("joint_carbon",700.,.90,.004,12.,.85),
    ("storage_only",700.,0.,.0002,6.,.65),
)
OBS_FIELDS=("base_load","mandatory_shore","flex_capacity","temperature","availability","weekly_progress","baseline_peak","policy_peak",
            "soc_deviation","accumulated_carbon","day_load_ratio","day_load_spread","weekday_sin","weekday_cos","month_sin","month_cos")


def fit_clock(env):
    # Every entry comes solely from the training segment. The raw V8 hourly
    # tariff and carbon factors repeat across days on this historical benchmark.
    values=env.segment.astype(np.float64);clock=np.arange(len(values))%24
    return {"profiles":[[float(v) for v in values[clock==h][:,[0,4,5]].mean(axis=0)] for h in range(24)],
            "scales":[env.load_center,env.load_std],"rows":len(values),"first":env.timestamps[0],"last":env.timestamps[-1]}


def forecast(env,clock):
    ctx=env._row_context();stamp=datetime.fromisoformat(env.timestamps[env._start+env._step].replace("Z","+00:00"))
    hours=(stamp.hour+np.arange(24))%24
    typical=np.array(clock["profiles"])[hours]
    # Anchor this day's load profile to the first current meter reading.
    ratio=ctx["base_load_kw"]/typical[0,0]
    load=typical[:,0]*ratio
    price=typical[:,1].copy();carbon=typical[:,2].copy()
    price[0]=ctx["price_cny_per_kwh"];carbon[0]=ctx["carbon_kg_per_kwh"]
    return load,price,carbon


def observation(env,clock):
    ctx=env._row_context();stamp=datetime.fromisoformat(env.timestamps[env._start+env._step].replace("Z","+00:00"))
    load,_,_=forecast(env,clock);center,scale=clock["scales"];week=stamp.weekday()*2*np.pi/7;year=stamp.month*2*np.pi/12
    return np.asarray([(ctx["base_load_kw"]-center)/scale,ctx["mandatory_shore_kw"]/8000,env.defer_limit_kw/200,
        (env._temperature_c-25)/15,ctx["equipment_availability_ratio"],env._step/168,(env.baseline_peak-center)/scale if env._step else 0,
        (env.totals["peak_kw"]-center)/scale if env._step else 0,(env._soc-env.soc_initial)*10,env.carbon_delta/150,
        load.mean()/center,np.ptp(load)/scale,np.sin(week),np.cos(week),np.sin(year),np.cos(year)],dtype=np.float32)


@lru_cache(maxsize=1)
def layout():
    # Each flexible-service variable connects a specific deferral hour to
    # a repayment hour no more than 12 hours later within the same day.
    src,dst=zip(*[(i,j) for i in range(24) for j in range(i+1,min(24,i+13))])
    src=np.array(src);dst=np.array(dst);n=49+len(src);ix=np.arange(24);arcs=48+np.arange(len(src))
    grid=np.zeros((24,n));grid[ix,ix]=1;grid[ix,24+ix]=-1
    grid[src,arcs]=-1;grid[dst,arcs]=1
    margin=grid.copy();margin[:,-1]=-1
    defer=np.zeros_like(grid);defer[src,arcs]=1
    repay=np.zeros_like(grid);repay[dst,arcs]=1
    return src,dst,grid,margin,defer,repay


def plan(env,clock,choice):
    name,power,share,reserve,shadow,peak_weight=OPTIONS[choice]
    load,price,carbon=forecast(env,clock)
    if name=="idle":return np.zeros((24,2),dtype=float)
    src,dst,grid,margin,defer,repay=layout();n=grid.shape[1]
    upper=np.full(n,np.inf);upper[:48]=power;cap=share*env.defer_limit_kw
    upper[48:-1]=cap
    if env._step+24==env.episode_steps:upper[48:-1][src>=12]=0.
    inventory=np.zeros((24,n));inventory[:,:24]=np.tri(24)*env.charge_eff
    inventory[:,24:48]=-np.tri(24)/env.discharge_eff
    # End each day neutral and respect native SOC envelopes at every hour.
    remaining=env.episode_steps-env._step-np.arange(1,25)
    band=np.minimum(.18*remaining/(env.episode_steps-1), remaining*env.power_kw*min(env.charge_eff,env.discharge_eff)/env.energy_kwh)
    positive=(np.minimum(env.soc_max,env.soc_initial+band)-env._soc)*env.energy_kwh
    negative=(env._soc-np.maximum(env.soc_min,env.soc_initial-band))*env.energy_kwh
    change=carbon@grid;throughput=np.ones(n);throughput[-1]=0.
    A=np.vstack([margin,grid,defer,repay,inventory,-inventory,change+reserve*throughput])
    rhs=np.r_[-load,env.hard_pcc_limit_kw-load,np.full(48,cap),positive,negative,0.]
    objective=(price+shadow*carbon)@grid
    objective[:48]+=env.cycle_cost
    rate=env.config["grid"]["demand_charge_cny_per_kw_month"]*env.episode_steps/(24*30.4375)
    objective[-1]=rate*peak_weight
    limits=np.column_stack([np.zeros(n),upper]);limits[-1,0]=max(0.,env.totals["peak_kw"])
    equality=np.array([(env.soc_initial-env._soc)*env.energy_kwh])
    opt=linprog(objective,A_ub=csr_matrix(A),b_ub=rhs,A_eq=csr_matrix(inventory[-1:]),b_eq=equality,bounds=limits,method="highs")
    if not opt.success:raise ValueError("Native asset daily option has no LP solution: "+opt.message)
    x=opt.x
    return np.stack([x[24:48]-x[:24],(repay-defer)@x],axis=1)


def coordinate(target,low,high,deadband):
    if low>high:return 0.
    neutral=float(np.clip(0.,low,high));target=float(np.clip(target,low,high))
    if abs(target-neutral)<1e-10:return 0.
    span=high-neutral if target>neutral else neutral-low
    return np.sign(target-neutral)*(deadband+(1-deadband)*abs(target-neutral)/span)


def dispatch(env,ctx,bess,flex):
    low,high=env._continuous_bounds(ctx)
    due=sum(amount for deadline,amount in env.flex_queue if deadline<=env._step)
    low=max(low,ctx["base_load_kw"]+due-env.hard_pcc_limit_kw)
    low,high=env._roundoff_interval(low,high,env.power_kw,env.energy_kwh/env.charge_eff,env.energy_kwh*env.discharge_eff)
    a=coordinate(bess,low,high,env.action_deadband);scheduled=env._coordinate(a,low,high)
    cap=ctx["auxiliary_shore_kw"]*env.flex_limit
    lo=-min(env.defer_limit_kw,cap,env.max_backlog_kwh-env._flex_backlog_kwh)
    if env.episode_steps-env._step-1<env.flex_deadline_hours:lo=0.
    if due>1e-9:lo=due
    hi=min(cap,env._flex_backlog_kwh,max(0.,env.hard_pcc_limit_kw-ctx["base_load_kw"]+scheduled))
    lo,hi=env._roundoff_interval(lo,hi,cap,env._flex_backlog_kwh)
    return np.array([a,coordinate(flex,lo,hi,env.action_deadband)],dtype=np.float64)


def run_day(env,clock,choice,trace=None):
    schedule=plan(env,clock,choice)
    money=carbon=reward=0.;last=None
    for h,(bess,flex) in enumerate(schedule):
        ctx=env._row_context();soc_before=env._soc
        if h==23:
            gap=(env._soc-env.soc_initial)*env.energy_kwh
            bess=gap*env.discharge_eff if gap>0 else gap/env.charge_eff
            flex=env._flex_backlog_kwh
        action=dispatch(env,ctx,bess,flex)
        _,r,done,trunc,last=env.step(action)
        assert not trunc and done==(env._step>=env.episode_steps)
        money+=last["v8_reward"]["bill_delta_cny"];carbon+=last["v8_reward"]["carbon_delta_kg"];reward+=r
        if trace is not None:
            trace.append({"hour":env._step-1,"option":OPTIONS[choice][0],"bess_kw":last["final_action"]["bess_kw"],
                  "flex_kw":last["final_action"]["flex_kw"],"base_kw":ctx["base_load_kw"],"pcc_kw":last["pcc_kw"],
                  "soc_before":soc_before,"soc":last["soc"],"price":ctx["price_cny_per_kwh"],"factor":ctx["carbon_kg_per_kwh"],
                  "mandatory_shore_kw":ctx["mandatory_shore_kw"],"auxiliary_kw":ctx["auxiliary_shore_kw"],
                  "service_short_kw":last["shore_sla_shortfall_kw"],"reserve_short_kw":last["reserve_shortfall_kw"],
                  "overdue_kwh":last["flex_overdue_kwh"],"backlog":env._flex_backlog_kwh,"guardrail":last["guardrail_violation"],
                  "requested_kw":[float(bess),float(flex)],"physical_violation":last["physical_power_violation"]})
    return {"cost_delta":money,"carbon_delta":carbon,"native_reward":reward,"day_end_soc":env._soc,
            "day_end_backlog":env._flex_backlog_kwh}
