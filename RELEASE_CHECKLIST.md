# Release checklist / 发布清单

- [ ] Version, changelog and citation metadata agree.
- [ ] Compile, the complete unit-test suite and seven-controller smoke test pass on a clean checkout.
- [ ] Secret scan and tracked-file review find no credentials, private paths or production telemetry.
- [ ] Dataset card, source URLs, licences and SHA-256 are current.
- [ ] README screenshots match the tagged build.
- [ ] No runtime run directory, model binary, audit secret, software-copyright or local environment artifact is tracked; portable evidence summaries are integrity checked.
- [ ] Container builds and `/health/live`, `/health/ready`, `/api/system/provenance` are checked.
- [ ] Security, execution and model-governance boundaries are unchanged or explicitly reviewed.
- [ ] Release notes identify replay/simulation/derived/measured data correctly.
- [ ] When made public: branch protection, private vulnerability reporting, discussions and security workflows are enabled.
