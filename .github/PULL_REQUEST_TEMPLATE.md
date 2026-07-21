## What changed / 变更内容

Describe the smallest reviewable change and the user or research problem it solves.

## Evidence / 验证证据

- [ ] `python -m compileall -q app scripts tests`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python -m scripts.rl_smoke_test --steps 64` (when RL/data/evaluation changes)
- [ ] UI or API behavior was checked when the visible workflow changes

## Data, model and safety impact / 数据、模型与安全影响

- Data provenance or schema impact:
- Training/evaluation boundary impact:
- Model-promotion or execution-gateway impact:
- Rollback plan:

## Disclosure / 声明

- [ ] No credentials, private port telemetry, vessel tracks or workstation paths are included.
- [ ] Simulated, derived, replayed and measured data remain explicitly distinguished.
- [ ] Third-party or generated assets are disclosed and redistributable.
