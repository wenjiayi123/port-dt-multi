"""Final r5 policy probe on the same three predeclared training windows.

Derived from train_actor_saturation_probe_20260912.py. No training, tuning or
policy evaluation outside the first 11664 historical TRAIN rows occurs here.
Only completed fixed-budget 60000-step models are accepted.
"""
from pathlib import Path
import sys,json,csv,itertools,hashlib,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()))
from stable_baselines3 import TD3,SAC
from app.services.rl_training.datasets import PortDataset,NUMERIC_COLUMNS,FACTOR_COLUMNS,file_sha256
from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
ROOT=Path.cwd(); torch.set_num_threads(1)
base=ROOT/'evidence/v8/shore_bess/runs'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--algorithm',choices=('sac','td3'),required=True);args=parser.parse_args()
names={'td3':'shore-bess-v8-td3-pilot-20260912-seed912-c16-physicalfix-r5','sac':'shore-bess-v8-sac-pilot-20260912-seed912-c16-physicalfix-r5'}
names={args.algorithm:names[args.algorithm]}
configs={k:json.loads((base/n/'config.json').read_text()) for k,n in names.items()}
ms={k:json.loads((base/n/'manifest.json').read_text()) for k,n in names.items()}
path=ROOT/'data/rl/datasets/public_cn_sha_hourly_v3.csv'
with path.open(encoding='utf-8-sig',newline='') as stream:
 rows=list(itertools.islice(csv.DictReader(stream),11664))
assert rows[-1]['timestamp']=='2025-04-30T23:00:00Z'
vals=np.asarray([[float(r[k]) for k in NUMERIC_COLUMNS] for r in rows],np.float32)
factors=np.asarray([[float(r[k]) if r.get(k) else 0 for k in FACTOR_COLUMNS] for r in rows],np.float32)
masks=np.asarray([[1.0 if r.get(k) else 0 for k in FACTOR_COLUMNS] for r in rows],np.float32)
data=PortDataset('public_cn_sha_hourly_v3',path,[r['timestamp'] for r in rows],vals,{'sha256':next(iter(ms.values()))['dataset_sha256']},factors,masks)
train=slice(0,len(rows)); starts=[1680,5040,8400]
result={'data_access':'CSV parser stopped after11664trainingrows; no validation or heldout rows parsed; no forward file opened','train_rows':len(rows),'train_starts':starts,'checkpoints':{},'source_sha256':{},'scenarios':{},'train_energy_profiles':[]}
for kind,name in names.items():
 m=ms[kind]
 completed=json.loads((base/name/'report.json').read_text())
 assert completed['total_environment_steps']==60000 and completed['config']['carbon_price_cny_per_kg_constraint_multiplier']==16.0
 assert completed['config']['n_step']==1 and completed['config']['algorithm_variant']=='vanilla_'+kind
 assert completed['config']==configs[kind] and completed['manifest']==m
 for filename in ['config.json','manifest.json','seed_912/curve.json']:
  referenced=base/name/filename
  assert file_sha256(referenced)==completed['evidence_files_sha256'][str(referenced.relative_to(ROOT))]
 result['source_report']={'path':str((base/name/'report.json').relative_to(ROOT)),'sha256':file_sha256(base/name/'report.json')}
 result['versions']=m['versions']
 for rel,sha in m['source_sha256'].items():
  assert file_sha256(ROOT/rel)==sha,(kind,rel)
  result['source_sha256'][rel]=sha
 curve_path=base/name/'seed_912/curve.json'
 anchors={r['step']:r['model_sha256'] for r in json.loads(curve_path.read_text())}
 for step in [60000]:
  ckpt=base/name/f'seed_912/step_{step}.zip'
  sha=file_sha256(ckpt); assert sha==anchors[step]
  assert sha==completed['evidence_files_sha256'][str(ckpt.relative_to(ROOT))]
  model=(TD3 if kind=='td3' else SAC).load(ckpt,device='cpu')
  assert model.num_timesteps==step
  key=f'{kind}_{step}'
  result['checkpoints'][key]={'path':str(ckpt.relative_to(ROOT)),'sha256':sha,'steps':int(model.num_timesteps),'sb3_update_counter':int(model._n_updates),'optimizer_step_calls':int(model._v8_optimizer_step_calls),'optimizer_steps_by_component':dict(model._v8_optimizer_steps_by_component)}
  totals=[]; acts=[]; logits=[]; qgrads=[]; actor_qgrads=[]; qs=[]; mcs=[]; rewards=[]; traces=[]
  for start in starts:
   cfg=configs[kind]
   env=ShoreBESSV8SACEnv(data,train,config=cfg['physical_config'],normalization_slice=train,episode_steps=168,carbon_price=cfg['carbon_price_cny_per_kg_constraint_multiplier'],training=False,record_trace=False)
   assert env.observation_space.shape==model.observation_space.shape==(31,)
   assert env.action_space.shape==model.action_space.shape==(2,)
   for field,value in cfg['normalization'].items():assert np.isclose(getattr(env,field),value,rtol=1e-12,atol=1e-12),(field,getattr(env,field),value)
   obs,_=env.reset(options={'start_index':start}); window=[]; contexts=[]
   for t in range(168):
    ot=torch.as_tensor(obs[None],dtype=torch.float32)
    with torch.no_grad():
     features=model.actor.extract_features(ot,model.actor.features_extractor)
     if kind=='td3':z=model.actor.mu[:-1](features)
     else:z=model.actor.mu(model.actor.latent_pi(features))
    action,_=model.predict(obs,deterministic=True)
    at=torch.as_tensor(action[None],dtype=torch.float32).clone().requires_grad_(True)
    values=model.critic(ot,at); q=torch.minimum(*values)
    q1grad=torch.autograd.grad(values[0].sum(),at,retain_graph=True)[0]
    actor_qgrad=torch.autograd.grad((values[0] if kind=='td3' else q).sum(),at,retain_graph=True)[0]
    actor_qgrads.append(actor_qgrad.detach().numpy()[0].tolist())
    obs,r,done,truncated,info=env.step(action)
    qvalue=float(q.detach().item())
    acts.append(action.tolist());logits.append(z.numpy()[0].tolist());qgrads.append(q1grad.detach().numpy()[0].tolist());qs.append(qvalue)
    window.append(r);rewards.append(r)
    contexts.append({'hour':t,'utc_hour':info['context']['timestamp_hour'],'price':info['context']['price_cny_per_kwh'],'carbon':info['context']['carbon_kg_per_kwh'],'bess_kw':info['final_action']['bess_kw'],'flex_kw':info['final_action']['flex_kw'],'action':action.tolist(),'qmin':qvalue,'logits':z.numpy()[0].tolist()})
   mcs.extend(np.cumsum(window[::-1])[::-1].tolist())
   totals.append({'start':start,**{k:env.totals[k] for k in ['financial_delta_cny','carbon_delta_kg','training_reward','physical_power_violations','flex_deadline_violation_kwh','terminal_flex_backlog_kwh','terminal_soc_error','reserve_shortfall_kwh','shore_sla_violation_kwh','max_flex_age_hours','guardrail_violation_rate','peak_kw','bess_throughput_kwh','aux_shift_kwh']}})
   traces.append({'start':start,'trace':contexts});env.close()
  a=np.asarray(acts);z=np.asarray(logits);grad=np.asarray(qgrads);qarr=np.asarray(qs);mc=np.asarray(mcs)
  result['scenarios'][key]={'windows':totals,'means':{k:float(np.mean([w[k] for w in totals])) for k in totals[0] if k!='start'},'action_min':a.min(0).tolist(),'action_max':a.max(0).tolist(),'action_mean':a.mean(0).tolist(),'saturation_abs_over_099':(np.abs(a)>.99).mean(0).tolist(),'exact_saturation_abs1':(np.abs(a)==1).mean(0).tolist(),'pre_tanh_min':z.min(0).tolist(),'pre_tanh_max':z.max(0).tolist(),'mean_tanh_jacobian':np.mean(1-a*a,axis=0).tolist(),'q1_action_grad_min':grad.min(0).tolist(),'q1_action_grad_max':grad.max(0).tolist(),'q1_action_grad_mean':grad.mean(0).tolist(),'chain_grad_mean_abs':np.mean(np.abs(grad*(1-a*a)),axis=0).tolist(),'qmin_mean':float(qarr.mean()),'qmin_range':[float(qarr.min()),float(qarr.max())],'monte_carlo_mean':float(mc.mean()),'qmin_minus_mc_mean':float((qarr-mc).mean()),'reward_mean_std':[float(np.mean(rewards)),float(np.std(rewards))],'trace':traces}
  result['scenarios'][key].update(pre_tanh_quantile_levels=[0,.01,.25,.5,.75,.99,1],pre_tanh_quantiles=np.quantile(z,[0,.01,.25,.5,.75,.99,1],axis=0).tolist(),tanh_jacobian_below_1e6=(1-a*a<1e-6).mean(0).tolist(),actor_objective_q_chain_grad_mean_abs=np.mean(np.abs(np.asarray(actor_qgrads)*(1-a*a)),axis=0).tolist(),gradient_claim_boundary='Q contribution through deterministic mean action only: Q1 for TD3, min(Q1,Q2) for SAC. Does not claim all actor parameter or SAC entropy gradients vanish.',monte_carlo_boundary='Same policy physical shaped rewards with gamma=1; SAC Q additionally includes entropy, so Q minus physical MC is diagnostic, not an exact value-error identity.')
for hour in range(24):
 subset=vals[np.arange(len(rows))%24==hour]
 result['train_energy_profiles'].append({'hour_utc':hour,'price_cny_kwh':float(np.mean(subset[:,4])),'carbon_kg_kwh':float(np.mean(subset[:,5]))})
result['probe_source_path']=str(Path(__file__).resolve().relative_to(ROOT));result['probe_source_sha256']=file_sha256(Path(__file__))
result['control_scope']='neural_bess_and_flexible_auxiliary_load';result['data_access']='CSV parser stops after 11664 TRAIN rows. Source file hashing may read whole CSV bytes, but no later rows are parsed or evaluated, and no forward file is loaded.'
result['tuning_or_training_performed']=False
out=ROOT/f'evidence/v8/shore_bess/audits/train-actor-saturation-r5-{args.algorithm}-20260912.json'
with out.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(out)
for key,r in result['scenarios'].items():print(key,json.dumps({k:v for k,v in r.items() if k not in ['trace','windows']}))
print('profile',result['train_energy_profiles'])
