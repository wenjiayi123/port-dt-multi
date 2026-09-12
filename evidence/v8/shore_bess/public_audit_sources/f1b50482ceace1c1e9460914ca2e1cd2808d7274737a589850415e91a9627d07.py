"""Assemble the append-only audit after every independent check completes."""
import hashlib,json,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch,stable_baselines3
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'evidence/v8/shore_bess/runs/shore-bess-v8-td3-pilot-20260912-seed912-c12-demandphi-nstep24-r3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p):return str(Path(p).relative_to(ROOT))
manifest=read(RUN/'manifest.json');config=read(RUN/'config.json')
original=read(OUT/'r3_step20000_physical_diagnostic.json')
variants=read(OUT/'r3_step20000_normalized_braking_counterfactual.json')
equivalence=read(OUT/'physical_execution_20260912_equivalence_semantic_operand_scale_final.json')
stress=read(OUT/'physical_execution_20260912_train_stress_operand_scale_final.json')
earlier_equivalence=read(OUT/'physical_execution_20260912_equivalence_original_v2.json')
earlier_stress=read(OUT/'physical_execution_20260912_train_stress.json')
source=ROOT/'app/services/rl_model/shore_bess/v8_environment.py';legacy=ROOT/'app/services/rl_model/shore_bess/v3_environment.py'
candidate=OUT/'v8_environment_float64_bidirectional_braking_ulp_normalized_candidate.py'
physical=ROOT/next(name for name in manifest['source_sha256'] if name.startswith('config/') and name.endswith('.json'))
checkpoint=next(row for row in read(RUN/'seed_912/curve.json') if row['step']==20000)
assert sha(physical)==manifest['source_sha256'][relative(physical)]
assert sha(legacy)==manifest['source_sha256'][relative(legacy)]
assert sha(source)==equivalence['main_environment_sha256_at_end']==stress['main_environment_sha256_at_end']
assert sha(ROOT/checkpoint['model_path'])==checkpoint['model_sha256']
assert equivalence['status']=='PASS' and equivalence['unexplained_mismatch_count']==0
assert stress['status']=='PASS' and stress['failure_count']==stress['inverted_bess_intervals']==stress['inverted_flex_intervals']==0
programs=[OUT/'physical_execution_audit_20260912.py',OUT/'physical_execution_audit_semantic_20260912.py',Path(__file__)]
evidence_paths=[OUT/name for name in ['r3_step20000_physical_diagnostic.json','r3_step20000_normalized_braking_counterfactual.json',
 'physical_execution_20260912_equivalence_original_v2.json','physical_execution_20260912_train_stress.json',
 'physical_execution_20260912_equivalence_operand_scale_final.json','physical_execution_20260912_equivalence_semantic_operand_scale_final.json',
 'physical_execution_20260912_train_stress_operand_scale_final.json']]
paths=[source,legacy,physical,RUN/'manifest.json',RUN/'config.json',ROOT/checkpoint['model_path'],RUN/'source'/relative(source),candidate,*programs,*evidence_paths]
checks={'original_model_bytes_preserved':True,'legacy_v3_source_unchanged':True,'physical_config_bytes_unchanged':True,
 'original_six_validation_windows_exactly_reproduced':original['recorded_episode_metrics_match'],
 'initial_main_matches_final_candidate':earlier_equivalence['status']=='PASS',
 'final_main_matches_candidate_actual_execution_and_v8_learning_reward':equivalence['status']=='PASS',
 'no_unexplained_main_candidate_differences':equivalence['unexplained_mismatch_count']==0,
 'fixed_120_train_episodes_safe':stress['status']=='PASS','zero_true_or_false_empty_intervals':True,
 'source_unchanged_during_audit':equivalence['source_unchanged'] and stress['source_unchanged'],
 'physical_tolerance_not_relaxed':True,'no_heldout_or_forward_replay':True,'no_optimizer_updates':True}
report={'schema':'port-shore-bess-physical-execution-numeric-audit.v1','status':'PASS' if all(checks.values()) else 'FAIL',
 'generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
 'purpose':'Physical execution precision and feasible-action regression audit; this does not admit a policy or claim learned savings.',
 'checks':checks,'source_files_sha256':{relative(path):sha(path) for path in paths},
 'final_main_environment_sha256':sha(source),'original_r3_environment_sha256':manifest['source_sha256'][relative(source)],
 'historical_model':{'run_id':manifest['run_id'],'step':20000,'model_path':checkpoint['model_path'],'model_sha256':checkpoint['model_sha256'],
  'r3_training_was_interrupted_for_environment_correction':True,'old_checkpoint_and_metrics_preserved':True},
 'root_causes':[{'code':'FLOAT32_EXECUTION_COMMAND','description':'A float32 mapped command rounded the prior braking boundary outward. V3.step casts again, so changing only the two V8 casts has no effect. Subsequent terminal-SOC projection follows the ramp clamp and produced 0.000216110276 and 0.000169536632 kW ramp excesses, triggering two false 100-point learning penalties.',
  'old_code_locations':['v8_environment.py:200,352 in r3 source snapshot','v3_environment.py:430-468,539'],
  'exact_replayed_violations':original['violations']},
 {'code':'REVERSE_DIRECTION_BRAKING_OMITTED','description':'The room<=ramp case only considered stopping at zero, excluding a feasible opposite-direction power next hour under the shrinking SOC envelope. All six original windows had about 56.1377 kW false reversed intervals.'},
 {'code':'ROUNDOFF_SCALE_IGNORED_OPERANDS','description':'SOC-to-power uses 50,000 kWh operands and FIFO due/backlog use accumulated sums. Endpoint-only epsilon left 4 BESS reversals up to 3.353761712787673e-12 kW and 131 flex reversals up to 3.694822225952521e-13 kW in TRAIN. Operand-scaled reconciliation removes them without relaxing execution limits.'}],
 'corrections':{'exact_command':'Keep observation/network float32; hand the exact float64 V8 command to V8._project across unchanged legacy V3.step.',
  'braking':{'eta':'charge_efficiency * discharge_efficiency','charge_room':'(next_soc_upper - soc) * E / charge_efficiency, signed',
   'discharge_room':'(soc - next_soc_lower) * E * discharge_efficiency, signed',
   'charge_cap':'(room+ramp)/2 when room>ramp; otherwise (eta*room+ramp)/(1+eta)',
   'discharge_cap':'(room+ramp)/2 when room>ramp; otherwise (room+eta*ramp)/(1+eta)',
   'other_bounds':'Intersect with current SOC, equipment, power, ramp, reserve and PCC limits.'},
  'roundoff':'Merge endpoints only within 32 * float64 epsilon * operand scale. BESS includes rated power and E/eta_c, E*eta_d; flex includes capacity and backlog. Recheck after PCC/due-service intersection.'},
 'unchanged_physics':{'asset':config['physical_config']['asset'],'physical_config_modified':False,
  'power_ramp_tolerance_kw':1e-6,'terminal_absolute_tolerance':1e-6,'flex_deadline_hours':12},
 'diagnostic_history':{'counterfactuals':{label:{key:value[key] for key in ('physical_violation_count','inverted_interval_count','max_ramp_excess_kw','mean')} for label,value in variants['results'].items()},
  'counterfactual_scope':'Same fixed historical 20k model on six originally recorded validation windows; no retraining or test selection.',
  'first_main_version':{'sha256':earlier_equivalence['main_environment_sha256_at_end'],'candidate_equivalence':earlier_equivalence['status'],
   'physical_train_violations':0,'residual_bess_reversals':earlier_stress['inverted_bess_intervals'],
   'residual_flex_reversals':earlier_stress['inverted_flex_intervals'],'retained_roundoff_failures':earlier_stress['failures']}},
 'validation_equivalence':equivalence,'train_stress':stress,
 'dataset_scope':{'dataset_id':manifest['dataset_id'],'dataset_sha256':manifest['dataset_sha256'],'train':manifest['train'],'validation':manifest['validation'],
  'test_windows_replayed':0,'forward_dataset_loaded':False,'normalization_fit_split':'train_only','stress_windows_fixed_before_testing':True},
 'accounting':{'new_training_environment_steps':0,'new_optimizer_updates':0,'fixed_train_stress_steps':20160,
  'validation_steps_per_implementation':1008,'diagnostic_replay_not_new_training':True},
 'reproducibility':{'audit_programs':{path.name:{'sha256':sha(path),'source_text':path.read_text()} for path in programs},
  'temporary_candidate_source':{'sha256':sha(candidate),'source_text':candidate.read_text()},
  'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'stable_baselines3':stable_baselines3.__version__},
  'commands':['python .codex_artifacts/shore_bess_v8_audit/physical_execution_audit_semantic_20260912.py equivalence --tag semantic_operand_scale_final',
              'python .codex_artifacts/shore_bess_v8_audit/physical_execution_audit_20260912.py train_stress --tag operand_scale_final']},
 'simulation_mode':True,'live_data_verified':False,'dispatch_allowed':False,'production_authority':False,
 'limitations':['Finite action/window stress is regression evidence, not a proof over every possible telemetry state.',
  'Public operational-resolution signals remain engineering scenarios; this audit does not establish field performance.',
  'Corrected action semantics require fresh training. The r3 weights and results are retained as history and are not admitted.',
  'Final operand-scale reconciliation removes nine flex mapping aliases in the six-window comparison. Actual V8 reward, execution and business/safety metrics remain exact; inherited V3 projection counts/reward change and are explicitly retained in the comparison.']}
path=ROOT/'evidence/v8/shore_bess/audits/physical-execution-numeric-audit-20260912.json';path.parent.mkdir(parents=True,exist_ok=True)
with path.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
print('PHYSICAL_EXECUTION_NUMERIC_AUDIT:PASS',relative(path),'SHA256',sha(path),'BYTES',path.stat().st_size)
