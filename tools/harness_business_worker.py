"""Isolated, source-pinned native V6 simulation workorder; never actuates site equipment."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import numpy as np
from app.services.rl_training.datasets import load_port_dataset
from app.services.rl_training.profiles import load_profile
from app.services.rl_training.coordinated_environment import CoordinatedPortOperationsEnv
from app.services.harness_event_execution import execute_schedule, interval_controls

def digest(data):
 return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def clean(data):
 if isinstance(data,dict): return {str(k):clean(v) for k,v in data.items()}
 if isinstance(data,(list,tuple,np.ndarray)): return [clean(v) for v in data]
 if isinstance(data,np.generic): return data.item()
 return data

def run(request):
 dataset=load_port_dataset('public_cn_sha_integrated_scenario_v5',ROOT/'data/rl/datasets')
 profile=load_profile('cn_sha_integrated_scenario_v5')
 files=[Path(__file__), ROOT/"app/services/harness_event_execution.py", *[(ROOT/'app/services/rl_training'/n) for n in ['environment.py','regulatory_environment.py','integrated_environment.py','coordinated_environment.py','datasets.py','profiles.py']]]
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 action=request.get('action','prepare')
 folder=Path(request['workorder_dir']).resolve()
 # Each workorder has an isolated directory; it cannot modify another or the live V3 runtime.
 allowed=(ROOT/'.runtime/harness-workorders').resolve()
 if folder.parent!=allowed or not folder.name.startswith('jm-'): raise ValueError('unregistered workorder directory')
 folder.mkdir(parents=True,exist_ok=True)
 plan_path=folder/'plan.json'; executed_path=folder/'executed.json'; rollback_path=folder/'rollback.json'
 controls=[0.]*18; controls[1]=1.
 controls[16]=min(1.,max(-1.,request.get('active_transport_vehicles',3)/5-.5))
 seed=44029
 def environment():
  env=CoordinatedPortOperationsEnv(dataset,slice(0,96),port_profile=profile,episode_steps=24,demand_cap_kw=profile["assets"]["demand_cap_kw"],seed=seed,training=False,record_trace=True,normalization_slice=slice(0,96))
  observation,info=env.reset(seed=seed,options={'start_index':0})
  return env,clean(observation),clean(info)
 env,initial,initial_info=environment()
 schedule=request.get('resource_schedule')
 event_execution=execute_schedule(schedule) if schedule else None
 control_schedule=interval_controls(schedule,24,env.step_hours,controls) if schedule else [controls for _ in range(24)]
 preview_env,_,_=environment()
 for control in control_schedule:
  _,_,terminated,truncated,_=preview_env.step(np.asarray(control,dtype=np.float32))
  if terminated or truncated:break
 predicted_totals=clean(preview_env.totals)
 if predicted_totals['violations'] > 0: raise ValueError('native preview violates the existing port profile constraints; no business plan admitted')
 plan={'schema':'native-port-workorder.v1','request_id':folder.name,'dataset_sha256':dataset.fingerprint,'profile_sha256':digest(profile),'sources':sources,
       'environment':'port_ops_v6','demand_cap_kw':profile['assets']['demand_cap_kw'],'seed':seed,'steps':24,'controls':controls,'control_schedule':control_schedule,'resource_schedule':schedule,'predicted_native_totals':predicted_totals,'initial_observation_sha256':digest(initial),
       'upstream':request.get('upstream',{}),'simulation_only':True,'production_authority':False,
       'scope':'native V6 numerical simulation: marine services, cargo move chain, gate, rail/barge, reefers, shore power and maintenance; regulatory release and site actuators are outside authority'}
 if action=='prepare':
  if plan_path.exists():
   if json.loads(plan_path.read_text())!=plan: raise ValueError('workorder already bound to different inputs or sources')
  else: plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
  return {'plan':plan,'plan_sha256':digest(plan),'status':'prepared'}
 if not plan_path.exists() or json.loads(plan_path.read_text())!=plan or request.get('plan_sha256')!=digest(plan): raise ValueError('native plan changed after approval')
 if action=='execute':
  if rollback_path.exists(): raise ValueError('workorder already rolled back; no replay')
  if executed_path.exists(): return json.loads(executed_path.read_text())
  if len(set(request.get('approved_by',[])))<2: raise ValueError('two distinct bound approvers required')
  rewards=[]
  for controls_at_step in plan['control_schedule']:
   observation,reward,terminated,truncated,info=env.step(np.asarray(controls_at_step,dtype=np.float32))
   rewards.append(float(reward))
   if terminated or truncated: break
  trace=clean(env.trace)
  if len(trace)!=24: raise ValueError('native business episode did not reach its full horizon')
  trace_path=folder/'execution-trace.json'; trace_path.write_text(json.dumps(trace,ensure_ascii=False,indent=2,allow_nan=False))
  result={'schema':'native-port-workorder-receipt.v1','status':'executed_simulation','plan_sha256':digest(plan),'simulation_only':True,'production_authority':False,
    'initial_observation_sha256':digest(initial),'final_observation_sha256':digest(clean(observation)), 'total_reward':sum(rewards),'steps':len(trace),
    'native_totals':clean(env.totals),'forecast_matches_execution':digest(clean(env.totals))==digest(predicted_totals),'resource_event_execution':event_execution,'trace_path':str(trace_path),'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    'approved_by':request['approved_by'],'approvals_simulated':bool(request.get('approvals_simulated')),'sources':sources,'dataset_sha256':dataset.fingerprint,
    'first_native_step':trace[0],'last_native_step':trace[-1]}
  executed_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
  (folder/'active-state.json').write_text(json.dumps({'state':'executed_simulation','receipt_sha256':digest(result)}))
  return result
 if action=='rollback':
  if not executed_path.exists(): raise ValueError('no executed workorder to reset')
  result={'status':'rolled_back_simulation','initial_observation_sha256':digest(initial),'restored_initial_state':digest(initial)==plan['initial_observation_sha256'],
    'execution_receipt_sha256':digest(json.loads(executed_path.read_text())),'scope':'reset this isolated native simulation only; executed trace retained; no reversal of external clock or physical cargo', 'production_authority':False}
  (folder/'active-state.json').write_text(json.dumps({'state':'initial','observation_sha256':digest(initial)}))
  rollback_path.write_text(json.dumps(result,indent=2))
  return result
 raise ValueError('unknown workorder action')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--result',required=True);args=parser.parse_args()
 result=run(json.loads(Path(args.request).read_text()))
 target=Path(args.result); temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temporary,target)
 print(result['status'])
