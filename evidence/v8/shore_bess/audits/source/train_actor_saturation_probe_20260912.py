from pathlib import Path
import sys,json,csv,itertools,hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()))
from stable_baselines3 import TD3,SAC
from app.services.rl_training.datasets import PortDataset,NUMERIC_COLUMNS,FACTOR_COLUMNS,file_sha256
from app.services.rl_model.shore_bess.v8_environment import ShoreBESSV8SACEnv
from scripts.train_shore_bess_v8_flex_specialist import ShoreBESSV8FlexSpecialistEnv
ROOT=Path.cwd(); torch.set_num_threads(1)
base=ROOT/'evidence/v8/shore_bess/runs'
names={'td3':'shore-bess-v8-td3-pilot-20260912-seed912-c12-nstep24-physicalfix-r4','sac_flex':'shore-bess-v8-flex-sac-diagnostic-20260912-seed912-c12-physicalfix-r4'}
configs={k:json.loads((base/n/'config.json').read_text()) for k,n in names.items()}
ms={k:json.loads((base/n/'manifest.json').read_text()) for k,n in names.items()}
path=ROOT/'data/rl/datasets/public_cn_sha_hourly_v3.csv'
with path.open(encoding='utf-8-sig',newline='') as stream:
 rows=list(itertools.islice(csv.DictReader(stream),11664))
assert rows[-1]['timestamp']=='2025-04-30T23:00:00Z'
vals=np.asarray([[float(r[k]) for k in NUMERIC_COLUMNS] for r in rows],np.float32)
factors=np.asarray([[float(r[k]) if r.get(k) else 0 for k in FACTOR_COLUMNS] for r in rows],np.float32)
masks=np.asarray([[1.0 if r.get(k) else 0 for k in FACTOR_COLUMNS] for r in rows],np.float32)
data=PortDataset('public_cn_sha_hourly_v3',path,[r['timestamp'] for r in rows],vals,{'sha256':ms['td3']['dataset_sha256']},factors,masks)
train=slice(0,len(rows)); starts=[1680,5040,8400]
result={'data_access':'CSV parser stopped after11664trainingrows; no validation or heldout rows parsed; no forward file opened','train_rows':len(rows),'train_starts':starts,'checkpoints':{},'source_sha256':{},'scenarios':{},'train_energy_profiles':[]}
for kind,name in names.items():
 m=ms[kind]
 for rel,sha in m['source_sha256'].items():
  if rel in ['app/services/rl_model/shore_bess/v8_environment.py','app/services/rl_model/shore_bess/v3_environment.py','scripts/train_shore_bess_v8_flex_specialist.py']:
   assert file_sha256(ROOT/rel)==sha,(kind,rel)
   result['source_sha256'][rel]=sha
 curve_path=base/name/('seed_912/curve.json' if kind=='td3' else 'curve.json')
 anchors={r['step']:r['model_sha256'] for r in json.loads(curve_path.read_text())}
 for step in ([10000,40000] if kind=='td3' else [30000]):
  ckpt=base/name/(f'seed_912/step_{step}.zip' if kind=='td3' else f'step_{step}.zip')
  sha=file_sha256(ckpt); assert sha==anchors[step]
  model=(TD3 if kind=='td3' else SAC).load(ckpt,device='cpu')
  assert model.num_timesteps==step
  key=f'{kind}_{step}'
  result['checkpoints'][key]={'path':str(ckpt.relative_to(ROOT)),'sha256':sha,'steps':int(model.num_timesteps),'updates':int(model._n_updates)}
  totals=[]; acts=[]; logits=[]; qgrads=[]; qs=[]; mcs=[]; rewards=[]; traces=[]
  for start in starts:
   cfg=configs[kind]
   env=(ShoreBESSV8SACEnv if kind=='td3' else ShoreBESSV8FlexSpecialistEnv)(data,train,config=cfg['physical_config'],normalization_slice=train,episode_steps=168,carbon_price=12.0,training=False,record_trace=False)
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
    obs,r,done,truncated,info=env.step(action)
    qvalue=float(q.detach().item())
    acts.append(action.tolist());logits.append(z.numpy()[0].tolist());qgrads.append(q1grad.detach().numpy()[0].tolist());qs.append(qvalue)
    window.append(r);rewards.append(r)
    contexts.append({'hour':t,'utc_hour':info['context']['timestamp_hour'],'price':info['context']['price_cny_per_kwh'],'carbon':info['context']['carbon_kg_per_kwh'],'bess_kw':info['final_action']['bess_kw'],'flex_kw':info['final_action']['flex_kw'],'action':action.tolist(),'qmin':qvalue,'logits':z.numpy()[0].tolist()})
   mcs.extend(np.cumsum(window[::-1])[::-1].tolist())
   totals.append({'start':start,**{k:env.totals[k] for k in ['financial_delta_cny','carbon_delta_kg','training_reward','physical_power_violations','flex_deadline_violation_kwh','terminal_flex_backlog_kwh','guardrail_violation_rate','peak_kw','bess_throughput_kwh','aux_shift_kwh']}})
   traces.append({'start':start,'trace':contexts});env.close()
  a=np.asarray(acts);z=np.asarray(logits);grad=np.asarray(qgrads);qarr=np.asarray(qs);mc=np.asarray(mcs)
  result['scenarios'][key]={'windows':totals,'means':{k:float(np.mean([w[k] for w in totals])) for k in totals[0] if k!='start'},'action_min':a.min(0).tolist(),'action_max':a.max(0).tolist(),'action_mean':a.mean(0).tolist(),'saturation_abs_over_099':(np.abs(a)>.99).mean(0).tolist(),'exact_saturation_abs1':(np.abs(a)==1).mean(0).tolist(),'pre_tanh_min':z.min(0).tolist(),'pre_tanh_max':z.max(0).tolist(),'mean_tanh_jacobian':np.mean(1-a*a,axis=0).tolist(),'q1_action_grad_min':grad.min(0).tolist(),'q1_action_grad_max':grad.max(0).tolist(),'q1_action_grad_mean':grad.mean(0).tolist(),'chain_grad_mean_abs':np.mean(np.abs(grad*(1-a*a)),axis=0).tolist(),'qmin_mean':float(qarr.mean()),'qmin_range':[float(qarr.min()),float(qarr.max())],'monte_carlo_mean':float(mc.mean()),'qmin_minus_mc_mean':float((qarr-mc).mean()),'reward_mean_std':[float(np.mean(rewards)),float(np.std(rewards))],'trace':traces}
for hour in range(24):
 subset=vals[np.arange(len(rows))%24==hour]
 result['train_energy_profiles'].append({'hour_utc':hour,'price_cny_kwh':float(np.mean(subset[:,4])),'carbon_kg_kwh':float(np.mean(subset[:,5]))})
out=Path('/tmp/shore-bess-v8-train-actor-probe.json');out.write_text(json.dumps(result,indent=2))
print(out)
for key,r in result['scenarios'].items():print(key,json.dumps({k:v for k,v in r.items() if k not in ['trace','windows']}))
print('profile',result['train_energy_profiles'])
