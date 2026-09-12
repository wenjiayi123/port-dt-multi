"""Independent numerical execution audit; original validation and fixed TRAIN windows only."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import TD3
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
OUT=Path(__file__).resolve().parent
RUN=ROOT/'evidence/v8/shore_bess/runs/shore-bess-v8-td3-pilot-20260912-seed912-c12-demandphi-nstep24-r3'
CANDIDATE=OUT/'v8_environment_float64_bidirectional_braking_ulp_normalized_candidate.py'
SOURCE=ROOT/'app/services/rl_model/shore_bess/v8_environment.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_candidate():
 spec=importlib.util.spec_from_file_location('physical_execution_final_candidate',CANDIDATE)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.ShoreBESSV8SACEnv

def config_data():
 config=json.loads((RUN/'config.json').read_text());manifest=json.loads((RUN/'manifest.json').read_text())
 data=load_port_dataset(manifest['dataset_id']);assert data.fingerprint==manifest['dataset_sha256']==sha(data.path)
 train=slice(manifest['train']['start_row'],manifest['train']['stop_row_exclusive']);val=slice(manifest['validation']['start_row'],manifest['validation']['stop_row_exclusive'])
 assert data.timestamps[train.stop-1]=='2025-04-30T23:00:00Z';assert data.timestamps[val.stop-1]=='2025-07-31T23:00:00Z'
 return config,manifest,data,train,val

def make_env(cls,config,data,train,split):
 return cls(data,split,config=config['physical_config'],normalization_slice=train,episode_steps=168,
  carbon_price=config['carbon_price_cny_per_kg_constraint_multiplier'],seed=0,training=False,record_trace=False)

def available(env,ctx):
 value=env.power_kw*ctx['equipment_availability_ratio']*max(.5,env._soh)
 if ctx['equipment_availability_ratio']<.5 or env._temperature_c>=env.temperature_trip_c:return 0.
 if env._temperature_c>env.temperature_derate_c:value*=(env.temperature_trip_c-env._temperature_c)/(env.temperature_trip_c-env.temperature_derate_c)
 return value

def equivalence():
 config,manifest,data,train,val=config_data();candidate=load_candidate()
 row=next(r for r in json.loads((RUN/'seed_912/curve.json').read_text()) if r['step']==20000)
 path=ROOT/row['model_path'];assert sha(path)==row['model_sha256']
 torch.set_num_threads(1);model=TD3.load(path,device='cpu');result=[];comparisons=0;failures=[]
 for start in row['evaluation']['starts']:
  environments=[make_env(cls,config,data,train,val) for cls in (ShoreBESSV8SACEnv,candidate)]
  states=[env.reset(options={'start_index':start})[0] for env in environments]
  for step in range(168):
   if not np.array_equal(states[0],states[1]):failures.append({'start':start,'step':step,'kind':'observation','max_error':float(np.max(np.abs(states[0]-states[1])))})
   predictions=[np.asarray(model.predict(obs,deterministic=True)[0]).astype(np.float32) for obs in states]
   if not np.array_equal(predictions[0],predictions[1]):failures.append({'start':start,'step':step,'kind':'neural_action'})
   outputs=[env.step(act) for env,act in zip(environments,predictions)]
   for name,values in [('reward',[v[1] for v in outputs]),('termination',[(v[2],v[3]) for v in outputs]),
                        ('final_action',[v[4]['final_action'] for v in outputs]),('mapping_values',[{k:value for k,value in v[4]['action_mapping'].items() if k!='version'} for v in outputs]),
                        ('guardrail',[v[4]['guardrail_violation'] for v in outputs]),('physical_flag',[v[4]['physical_power_violation'] for v in outputs]),
                        ('ledger',[env.totals for env in environments])]:
    if values[0]!=values[1]:failures.append({'start':start,'step':step,'kind':name})
   states=[v[0] for v in outputs];comparisons+=1
  result.append({'start_index':start,'first_timestamp':environments[0].timestamps[start],
                 'last_timestamp':environments[0].timestamps[start+167],'main_metrics':environments[0].totals,
                 'candidate_metrics':environments[1].totals})
  for env in environments:env.close()
 return {'status':'PASS' if not failures else 'FAIL','checkpoint_path':str(path.relative_to(ROOT)),
         'checkpoint_sha256':sha(path),'checkpoint_steps':20000,'windows':row['evaluation']['starts'],
         'compared_environment_steps':comparisons,'comparison':'exact equality of float32 observation/neural action and every step reward, numeric mapping, execution, ledger and flags; mapping version label is separately recorded',
         'mapping_version_labels':{'main':ShoreBESSV8SACEnv.action_mapping_version,'candidate':candidate.action_mapping_version},
         'mismatch_count':len(failures),'mismatches':failures,'per_window':result}

def train_stress():
 config,manifest,data,train,_=config_data()
 weeks=list(range(0,train.stop-train.start-168,168));starts=[weeks[int(i)] for i in np.linspace(0,len(weeks)-1,20,dtype=int)]
 patterns=['saturated_charge_defer','saturated_charge_repay','saturated_discharge_defer','saturated_discharge_repay','alternating_corners','random_uniform']
 constants={'saturated_charge_defer':[-1,-1],'saturated_charge_repay':[-1,1],
            'saturated_discharge_defer':[1,-1],'saturated_discharge_repay':[1,1]}
 rows=[];failures=[];steps=0;max_ramp=-float('inf');max_available=-float('inf');inversions=0;flex_inversions=0
 for pattern in patterns:
  for start in starts:
   rng=np.random.default_rng(20260912+start);env=make_env(ShoreBESSV8SACEnv,config,data,train,train)
   obs,_=env.reset(options={'start_index':start})
   for step in range(168):
    assert obs.dtype==np.float32
    action=np.asarray(constants[pattern] if pattern in constants else ([-1,-1] if step%2==0 else [1,1]) if pattern=='alternating_corners' else rng.uniform(-1,1,2),dtype=np.float32)
    ctx=env._row_context();power_before=env._last_bess_kw;limit=available(env,ctx)
    previous={'power_kw':power_before,'soc':env._soc,'soh':env._soh,'temperature_c':env._temperature_c}
    obs,reward,terminated,truncated,info=env.step(action);steps+=1
    actual=info['final_action']['bess_kw'];ramp=abs(actual-power_before)-env.ramp_kw;available_excess=abs(actual)-limit
    max_ramp=max(max_ramp,ramp);max_available=max(max_available,available_excess)
    mapping=info['action_mapping'];lo,hi=mapping['bess_interval_kw'];flo,fhi=mapping['flex_interval_kw'];inversions+=int(lo>hi);flex_inversions+=int(flo>fhi)
    reasons=[]
    if lo>hi:reasons.append('empty_bess_interval')
    if flo>fhi:reasons.append('empty_flex_interval')
    if ramp>1e-6:reasons.append('independent_ramp_excess')
    if available_excess>1e-6:reasons.append('independent_available_power_excess')
    if info['physical_power_violation']:reasons.append('environment_physical_violation')
    if info['guardrail_violation']:reasons.append('guardrail_violation')
    if info['flex_overdue_kwh']>1e-6:reasons.append('deadline_overdue')
    if not env.soc_min-1e-9<=env._soc<=env.soc_max+1e-9:reasons.append('soc_outside_limits')
    if reasons:failures.append({'pattern':pattern,'window_start':start,'step':step,'timestamp':env.timestamps[start+step],
          'reasons':reasons,'previous':previous,'action':action.tolist(),'mapping':mapping,'final_action':info['final_action'],
          'ramp_excess_kw':ramp,'available_excess_kw':available_excess})
   totals=env.totals
   for field in ('physical_power_violations','guardrail_violation_rate','terminal_soc_error','terminal_flex_backlog_kwh','flex_deadline_violation_kwh','shore_sla_violation_kwh','reserve_shortfall_kwh'):
    if totals[field]>1e-6:failures.append({'pattern':pattern,'window_start':start,'kind':'episode_ledger','field':field,'value':totals[field]})
   rows.append({'pattern':pattern,'start_index':start,'first_timestamp':env.timestamps[start],'last_timestamp':env.timestamps[start+167],
                'metrics':{k:v for k,v in totals.items() if any(s in k for s in ('violation','terminal','reserve','age','soc_','temperature'))}})
   env.close()
  print('completed',pattern,'windows',len(starts),'failures_so_far',len(failures),flush=True)
 return {'status':'PASS' if not failures else 'FAIL','partition':'TRAIN only, 2024-01-01 through 2025-04-30',
         'window_selection':'20 evenly spaced indices from 69 complete nonoverlapping train weeks, chosen before running any stress action',
         'starts':starts,'patterns':patterns,'random_seed_rule':'numpy default_rng(20260912 + start_index)',
         'episodes':len(rows),'environment_steps':steps,'maximum_ramp_excess_kw':max_ramp,
         'maximum_available_power_excess_kw':max_available,'inverted_bess_intervals':inversions,'inverted_flex_intervals':flex_inversions,
         'violation_tolerance_kw':1e-6,'failure_count':len(failures),'failures':failures,'per_episode':rows}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['equivalence','train_stress']);parser.add_argument('--tag',default='refined');args=parser.parse_args()
 initial=sha(SOURCE);value=equivalence() if args.mode=='equivalence' else train_stress()
 value.update(main_environment_sha256_at_start=initial,main_environment_sha256_at_end=sha(SOURCE),source_unchanged=initial==sha(SOURCE))
 path=OUT/f'physical_execution_20260912_{args.mode}_{args.tag}.json';path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
 print(args.mode,value['status'],'OUTPUT',path,flush=True)
if __name__=='__main__':main()
