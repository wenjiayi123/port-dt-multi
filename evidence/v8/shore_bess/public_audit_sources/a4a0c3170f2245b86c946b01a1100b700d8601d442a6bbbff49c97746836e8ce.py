"""Additive completion audits. Never edit original reports or training outputs."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(run_id, log_name, audit_name):
    directory = ROOT / 'evidence/v8/shore_bess/runs' / run_id
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    log = ROOT / '.codex_artifacts/shore_bess_v8_audit' / log_name
    raw = log.read_text()
    assert raw.endswith('SHORE_BESS_V8_INTEGRITY:PASS\n')
    verification = json.loads(raw.split('\nSHORE_BESS_V8_INTEGRITY:')[0])
    assert verification['integrity_status'] == 'PASS'
    assert verification['report_sha256'] == sha(report_path)
    assert verification['current_source_drift'] == []
    result = report['results'][0]
    seed = directory / f"seed_{result['seed']}"
    curve = json.loads((seed / 'curve.json').read_text())
    final = curve[-1]
    assert final['step'] == 60000
    assert verification['replay']['final_validation']['checkpoint_step'] == final['step']
    assert verification['replay']['final_validation']['model_sha256'] == final['model_sha256']
    with (seed / 'training_episodes.csv').open(newline='') as handle:
        ledger = list(csv.DictReader(handle))
    typed_ledger = []
    for row in ledger:
        typed_ledger.append({key: value if key == 'record_phase' else float(value) if key.startswith('episode_metrics.') else int(value)
                             for key, value in row.items()})
    config = report['config']
    counterfactuals = []
    for row in curve:
        counterfactuals.append({key: row[key] for key in ('step', 'optimizer_updates', 'sb3_update_counter', 'weights_sha256',
            'model_path', 'model_sha256', 'comparison', 'gates')})
    anchored = {str(path.relative_to(ROOT)): sha(path) for path in
                (report_path, seed / 'training_episodes.csv', seed / 'monitor.csv', seed / 'curve.json',
                 directory / 'config.json', directory / 'manifest.json', directory / 'selection.json',
                 ROOT / 'scripts/verify_shore_bess_v8.py', Path(__file__).resolve())}
    audit = {
        'schema': 'shore-bess-v8-independent-completion-audit.v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run_id': run_id,
        'scope': 'Independent integrity, physical training ledger accounting, recorded validation statistics and selected/final validation model replay. No new training and no new held-out policy evaluation.',
        'integrity_status': 'PASS',
        'reported_status': report['status'],
        'admitted_offline': verification['admitted_offline'],
        'original_report_preserved': True,
        'simulation_mode': True, 'live_data_verified': False, 'dispatch_allowed': False, 'production_authority': False,
        'anchored_files_sha256': anchored,
        'training_source_sha256': report['manifest']['source_sha256'],
        'algorithm': config['algorithm'],
        'algorithm_variant': config['algorithm_variant'],
        'carbon_constraint_multiplier_cny_per_kg': config['carbon_price_cny_per_kg_constraint_multiplier'],
        'seed': result['seed'],
        'actual_environment_steps': result['steps'],
        'actual_optimizer_step_calls': result['optimizer_updates'],
        'actual_optimizer_step_calls_by_component': result['parameters']['optimizer_step_calls_by_component'],
        'sb3_gradient_iterations': result['sb3_update_counter'],
        'independent_verification': verification,
        'training_episode_ledger': {
            'descriptor': result['training_episode_ledger'],
            'rows': typed_ledger,
            'scope': 'All 357 completed 168-hour training episodes (59,976 steps); the last 24 steps are partial and have no completed-episode physical metrics. Training windows may repeat, so their sample means are descriptive, not independent held-out estimates.',
            'audit_summary': verification['training_episode_ledgers'][0],
        },
        'checkpoint_curve': counterfactuals,
        'distinct_checkpoint_weight_hashes': len({row['weights_sha256'] for row in curve}),
        'selected_checkpoint_step': result['selected']['step'],
        'fixed_final_checkpoint_step': final['step'],
        'selected_checkpoint_mean_gains_percent': {key: result['selected']['comparison'][key]['mean'] for key in ('total_cost_cny', 'carbon_kg', 'peak_kw')},
        'fixed_final_mean_gains_percent': {key: final['comparison'][key]['mean'] for key in ('total_cost_cny', 'carbon_kg', 'peak_kw')},
        'fixed_final_every_window_carbon_improved': all(value > 0 for value in final['comparison']['carbon_kg']['per_window_percent']),
        'convergence': result['convergence'],
        'final_replay_buffer_snapshot': result['parameters'].get('replay_buffer_snapshot'),
        'interpretation': 'The candidate did perform neural-network optimizer updates but did not pass the fixed business/admission criteria. Integrity PASS is not admission. Zero violations in recorded complete training episodes do not establish field safety or unrecorded partial-episode safety.',
        'held_out_policy_evaluation_partitions': list(report['evaluations']),
        'new_training_environment_steps_executed_by_this_audit': 0,
        'new_optimizer_updates_executed_by_this_audit': 0,
        'limitations': [
            'SHA anchors establish internal consistency without independent signed wall-clock attestation.',
            'The loaded historical dataset includes the existing train/validation/repeated-test CSV for hash and source checks; only already-recorded validation policies were replayed here. No new test or forward policy evaluation was performed.',
            'The 24-step replay variant has no off-policy importance correction and is not vanilla TD3. Final storage-only flush does not imply the newly stored tails were sampled or learned.'
            if config['n_step'] > 1 else 'This is standard one-step TD3; it is not a multi-step TD3 experiment.',
        ],
    }
    destination = ROOT / 'evidence/v8/shore_bess/audits' / audit_name
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    assert sha(report_path) == verification['report_sha256']
    for relative, expected in anchored.items():
        assert sha(ROOT / relative) == expected
    print(json.dumps({'path': str(destination.relative_to(ROOT)), 'sha256': sha(destination),
                      'actual_training_steps': result['steps'], 'actual_optimizer_calls': result['optimizer_updates'],
                      'complete_episodes': len(ledger), 'unsafe_complete_episodes': len(verification['training_episode_ledgers'][0]['unsafe_completed_episode_indices']),
                      'selected_step': result['selected']['step'], 'final_gains': audit['fixed_final_mean_gains_percent']}))

build('shore-bess-v8-td3-pilot-20260912-seed912-c12-nstep24-physicalfix-r4',
      'verifier_td3_nstep24_r4_with_ledger_final.log', 'td3-nstep24-r4-independent-audit-20260912.json')
build('shore-bess-v8-td3-pilot-20260912-seed912-c16-physicalfix-r5',
      'verifier_td3_r5_with_ledger.log', 'td3-standard-r5-independent-audit-20260912.json')
