from datetime import datetime
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,vstack
from settlement_options import MENU,matrix,forecast

def plan(env,clock,choice,h,cap_kg,grid_change_cap=None):
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
    # Additional independent carbon ceiling versus native actions at this epoch.
    A=vstack([A,csr_matrix(co2.reshape(1,-1))],format="csr")
    rhs=np.r_[rhs,cap_kg]
    if grid_change_cap is not None:
        A=vstack([A,csr_matrix(g)],format='csr')
        rhs=np.r_[rhs,grid_change_cap]
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
