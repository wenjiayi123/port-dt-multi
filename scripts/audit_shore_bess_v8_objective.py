#!/usr/bin/env python3
"""TRAIN-only scalar-objective LP diagnostic; perfect-foresight relaxation, never RL.

Only the pinned first 11,664 CSV data rows are parsed. Validation/test/forward
observations never enter this diagnostic. Peak increases and carbon increases
are permitted and charged by the actual bill / explicit carbon multiplier.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,itertools,json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from app.services.rl_training.datasets import PortDataset,NUMERIC_COLUMNS,FACTOR_COLUMNS
from app.services.rl_model.shore_bess.v3_environment import load_config,fixed_window_starts
from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
TRAIN_ROWS=11664
LAMBDAS=(8.,12.,16.,20.,30.,40.)
HOURS=168

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_training_prefix():
    path=ROOT/'data/rl/datasets/public_cn_sha_hourly_v3.csv'
    with path.open('rb') as stream:
        lines=list(itertools.islice(stream,TRAIN_ROWS+1))
    assert len(lines)==TRAIN_ROWS+1
    prefix=b''.join(lines);prefix_sha=hashlib.sha256(prefix).hexdigest()
    records=list(csv.DictReader(io.StringIO(prefix.decode('utf-8-sig'))))
    timestamps=[row['timestamp'] for row in records]
    expected=[(datetime(2024,1,1,tzinfo=timezone.utc)+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M:%SZ') for i in range(TRAIN_ROWS)]
    assert timestamps==expected and timestamps[-1]=='2025-04-30T23:00:00Z'
    values=np.asarray([[float(row[name]) for name in NUMERIC_COLUMNS] for row in records],dtype=np.float32)
    factors=np.asarray([[float(row[name]) if row.get(name) else 0. for name in FACTOR_COLUMNS] for row in records],dtype=np.float32)
    masks=np.asarray([[float(bool(row.get(name))) for name in FACTOR_COLUMNS] for row in records],dtype=np.float32)
    assert np.isfinite(values).all() and np.isfinite(factors).all()
    dataset=PortDataset('public_cn_sha_hourly_v3_train_prefix_only',path,timestamps,values,
       {'sha256':prefix_sha,'rows':TRAIN_ROWS,'fingerprint_scope':'exact CSV header plus first 11,664 data rows only'},factors,masks)
    return dataset,{'path':str(path.relative_to(ROOT)),'train_prefix_sha256':prefix_sha,'parsed_data_rows':TRAIN_ROWS,
                    'first_timestamp':timestamps[0],'last_timestamp':timestamps[-1],
                    'feature_dtype':'float32 as production loader, then float64 in LP',
                    'validation_rows_parsed':0,'test_rows_parsed':0,'forward_dataset_loaded':False}

def problem(dataset,start,bess_enabled):
    train=slice(0,dataset.rows)
    env=ShoreBESSV8SACEnv(dataset,train,config=load_config(),normalization_slice=train,
                         episode_steps=HOURS,training=False,seed=0,carbon_price=12.)
    env.reset(options={'start_index':start});contexts=[]
    for step in range(HOURS):
        env._step=step;contexts.append(env._row_context())
    def col(name):return np.asarray([ctx[name] for ctx in contexts],dtype=np.float64)
    base,price,carbon=col('base_load_kw'),col('price_cny_per_kwh'),col('carbon_kg_per_kwh')
    aux_capacity=col('auxiliary_shore_kw')*env.flex_limit
    available=env.power_kw*col('equipment_availability_ratio')*env.soh_initial
    available[col('equipment_availability_ratio')<.5]=0.
    n=5*HOURS+1
    def ix(block,t):return block*HOURS+t
    bounds=[]
    bounds.extend((0.,max(0.,min(available[t],env.hard_pcc_limit_kw-base[t])) if bess_enabled else 0.) for t in range(HOURS))
    bounds.extend((0.,max(0.,available[t]-contexts[t]['reserve_required_kw']) if bess_enabled else 0.) for t in range(HOURS))
    bounds.extend((-min(env.defer_limit_kw,float(cap)) if HOURS-t-1>=env.flex_deadline_hours else 0.,float(cap)) for t,cap in enumerate(aux_capacity))
    for t in range(HOURS):
        remaining=HOURS-t-1
        band=min(remaining*env.power_kw*min(env.charge_eff,env.discharge_eff)/env.energy_kwh,.18*remaining/(HOURS-1))
        bounds.append((max(env.soc_min,env.soc_initial-band),min(env.soc_max,env.soc_initial+band)))
    bounds.extend((0.,min(env.max_backlog_kwh,float(aux_capacity[t+1:].sum()))) for t in range(HOURS))
    # Actual demand charge supplies the incentive. No peak-non-regression guard.
    bounds.append((0.,env.hard_pcc_limit_kw))
    eq=[];eqrhs=[];ub=[];ubrhs=[]
    for t in range(HOURS):
        row={ix(3,t):1.,ix(0,t):-env.charge_eff/env.energy_kwh,ix(1,t):1./(env.discharge_eff*env.energy_kwh)}
        if t:row[ix(3,t-1)]=-1.
        eq.append(row);eqrhs.append(env.soc_initial if t==0 else 0.)
        row={ix(4,t):1.,ix(2,t):1.}
        if t:row[ix(4,t-1)]=-1.
        eq.append(row);eqrhs.append(0.)
        flow={ix(0,t):1.,ix(1,t):-1.,ix(2,t):1.}
        ub.append({**flow,n-1:-1.});ubrhs.append(-float(base[t]))
        ub.append({i:-v for i,v in flow.items()});ubrhs.append(float(base[t]))
        ramp={ix(1,t):1.,ix(0,t):-1.}
        if t:ramp.update({ix(1,t-1):-1.,ix(0,t-1):1.})
        ub.extend([ramp,{i:-v for i,v in ramp.items()}]);ubrhs.extend([env.ramp_kw,env.ramp_kw])
    def matrix(rows):
        a=lil_matrix((len(rows),n))
        for j,row in enumerate(rows):
            for i,v in row.items():a[j,i]=v
        return a.tocsr()
    cost=np.zeros(n);co2=np.zeros(n)
    cost[:HOURS]=price+env.cycle_cost;cost[HOURS:2*HOURS]=-price+env.cycle_cost;cost[2*HOURS:3*HOURS]=price
    co2[:HOURS]=carbon;co2[HOURS:2*HOURS]=-carbon;co2[2*HOURS:3*HOURS]=carbon
    rate=float(env.config['grid']['demand_charge_cny_per_kw_month'])*HOURS/(24.*30.4375);cost[-1]=rate
    result={'A_eq':matrix(eq),'b_eq':np.asarray(eqrhs),'A_ub':matrix(ub),'b_ub':np.asarray(ubrhs),'bounds':bounds,
            'cost':cost,'carbon':co2,'base':base,'price':price,'carbon_factor':carbon,'demand_rate':rate,
            'baseline_cost':float(base@price+rate*base.max()),'baseline_carbon':float(base@carbon),
            'defer_limit_kw':env.defer_limit_kw,'max_backlog_kwh':env.max_backlog_kwh,'aux_capacity':aux_capacity,
            'charge_efficiency':env.charge_eff,'discharge_efficiency':env.discharge_eff,'energy_kwh':env.energy_kwh}
    env.close();return result

def solve(data,start,bess_enabled,lam,p):
    c=p['cost']+lam*p['carbon']
    result=linprog(c,A_eq=p['A_eq'],b_eq=p['b_eq'],A_ub=p['A_ub'],b_ub=p['b_ub'],bounds=p['bounds'],method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10})
    if not result.success:raise RuntimeError(f'LP failed: start={start},bess={bess_enabled},lambda={lam}: {result.message}')
    x=result.x;charge,discharge,flex,soc,backlog=[x[i*HOURS:(i+1)*HOURS] for i in range(5)]
    net=p['base']+charge-discharge+flex;peak=float(net.max());basepeak=float(p['base'].max())
    total_cost=float(net@p['price']+.08*(charge+discharge).sum()+p['demand_rate']*peak)
    carbon=float(net@p['carbon_factor']);cost_delta=total_cost-p['baseline_cost'];carbon_delta=carbon-p['baseline_carbon']
    objective=cost_delta+lam*carbon_delta
    objective_lp=float(result.fun-p['demand_rate']*basepeak)
    assert abs(objective-objective_lp)<1e-5
    equality_error=float(np.max(np.abs(p['A_eq']@x-p['b_eq'])))
    inequality_excess=max(0.,float(np.max(p['A_ub']@x-p['b_ub'])))
    assert equality_error<1e-6 and inequality_excess<1e-6
    simultaneous=float(np.minimum(charge,discharge).sum())
    return {'start_index':start,'first_timestamp':data.timestamps[start],'last_timestamp':data.timestamps[start+HOURS-1],
        'mode':'full_bess_and_flex' if bess_enabled else 'flex_only','lambda_cny_per_kg':lam,
        'cost_reduction_percent':-100.*cost_delta/p['baseline_cost'],'carbon_reduction_percent':-100.*carbon_delta/p['baseline_carbon'],
        'peak_reduction_percent':100.*(basepeak-peak)/basepeak,'cost_delta_cny':cost_delta,'carbon_delta_kg':carbon_delta,
        'peak_delta_kw':peak-basepeak,'weighted_objective_delta_cny':objective,'weighted_objective_improvement_cny':-objective,
        'weighted_carbon_component_cny':lam*carbon_delta,'baseline_cost_cny':p['baseline_cost'],'baseline_carbon_kg':p['baseline_carbon'],
        'baseline_peak_kw':basepeak,'candidate_cost_cny':total_cost,'candidate_carbon_kg':carbon,'candidate_peak_kw':peak,
        'baseline_demand_charge_cny':p['demand_rate']*basepeak,'candidate_demand_charge_cny':p['demand_rate']*peak,
        'charge_kwh':float(charge.sum()),'discharge_kwh':float(discharge.sum()),'flex_shift_kwh':float(np.abs(flex).sum()/2.),
        'terminal_soc':float(soc[-1]),'terminal_backlog_kwh':float(backlog[-1]),'maximum_backlog_kwh':float(backlog.max()),
        'simultaneous_charge_discharge_kwh':simultaneous,'primal_equality_max_abs_error':equality_error,
        'primal_inequality_max_excess':inequality_excess,'solver_status':int(result.status),'solver_message':result.message,
        'schedule':{'charge_kw':charge.tolist(),'discharge_kw':discharge.tolist(),'flex_kw':flex.tolist(),'soc_after':soc.tolist(),
                    'backlog_after_kwh':backlog.tolist(),'grid_kw':net.tolist()}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'evidence/v8/shore_bess/audits/train-scalar-objective-audit-20260912.json');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Do not overwrite a retained audit')
    data,scope=load_training_prefix();starts=fixed_window_starts(data.rows,HOURS,6)
    assert starts==[0,2299,4598,6897,9196,11495]
    source_paths=[Path(__file__),ROOT/'app/services/rl_model/shore_bess/v8_environment.py',ROOT/'app/services/rl_model/shore_bess/v3_environment.py',
                  ROOT/'app/services/rl_training/datasets.py',ROOT/'config/shore_bess_v3.json']
    sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths};rows=[];configuration={}
    for enabled in (True,False):
        for start in starts:
            p=problem(data,start,enabled)
            configuration={'defer_limit_kw':p['defer_limit_kw'],'max_backlog_kwh':p['max_backlog_kwh']}
            for lam in LAMBDAS:
                row=solve(data,start,enabled,lam,p);rows.append(row)
        print('completed',('full BESS+flex' if enabled else 'flex-only'),'36 LP solves',flush=True)
    summaries=[]
    for mode in ('full_bess_and_flex','flex_only'):
        for lam in LAMBDAS:
            group=[r for r in rows if r['mode']==mode and r['lambda_cny_per_kg']==lam]
            metrics={key:{'mean':float(np.mean([r[key] for r in group])),'minimum':float(min(r[key] for r in group)),
                      'maximum':float(max(r[key] for r in group))} for key in ('cost_reduction_percent','carbon_reduction_percent','peak_reduction_percent',
                      'cost_delta_cny','carbon_delta_kg','peak_delta_kw','weighted_objective_delta_cny','weighted_objective_improvement_cny')}
            summary={'mode':mode,'lambda_cny_per_kg':lam,'window_count':len(group),'carbon_increasing_window_count':sum(r['carbon_delta_kg']>1e-6 for r in group),
                     'cost_increasing_window_count':sum(r['cost_delta_cny']>1e-6 for r in group),'metrics':metrics}
            summaries.append(summary);print(json.dumps(summary),flush=True)
    _,final_scope=load_training_prefix();assert final_scope==scope
    assert all(digest(ROOT/name)==sha for name,sha in sources.items())
    report={'schema':'port-shore-bess-v8-train-scalar-objective-audit.v1','generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
       'scope':'TRAIN only, optimistic perfect-foresight linear-program diagnostic','is_neural_policy':False,'is_learned_policy_result':False,
       'dataset':scope,'window_hours':HOURS,'fixed_window_starts':starts,'lambdas_cny_per_kg':list(LAMBDAS),
       'objective':'minimize actual electricity + degradation + demand cost delta + lambda * carbon delta relative to idle baseline',
       'carbon_non_regression_constraint':False,'cost_non_regression_constraint':False,'peak_non_regression_constraint':False,
       'source_sha256':sources,'source_and_training_prefix_unchanged':True,'physical_config':load_config(),'flex_parameters':configuration,
       'assumptions':['Perfect knowledge of each complete training window, noncausal LP oracle.',
         'Original physical asset, charge/discharge efficiencies, ramp, PCC, reserve and demand-charge accounting.',
         'Reserve hours use current V8 Asia/Shanghai conversion; tariff/carbon remain the pinned UTC engineering scenario.',
         'Current V8 train-derived defer cap and last-12h no-new-deferral rule retained; no prepayment, SOC envelope and final inventory closure retained.',
         'Constant initial SOH and no temperature nonlinearities are optimistic relaxations.',
         'Maximum FIFO deferral age omitted: resulting LP optimum is not a deployable or admitted V8 policy.',
         'No direct LP carbon/cost/peak non-regression constraint is imposed; increases are allowed and explicitly reported.',
         'Separate charge/discharge variables are continuous; any simultaneous activity is explicitly reported.'],
       'summaries':summaries,'windows':rows,'solver':{'name':'scipy.optimize.linprog/HiGHS','scipy_version':scipy.__version__,
         'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10},
       'new_training_environment_steps':0,'new_optimizer_updates':0,'simulation_mode':True,'live_data_verified':False,'dispatch_allowed':False,'production_authority':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print('TRAIN_SCALAR_OBJECTIVE_AUDIT:PASS',args.output.relative_to(ROOT),'SHA256',digest(args.output),flush=True)
if __name__=='__main__':main()
