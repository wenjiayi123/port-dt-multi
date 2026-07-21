# Security Policy / 安全策略

## Reporting / 漏洞报告

Do not disclose port topology, device addresses, credentials, non-public vessel tracks, production telemetry or exploitable details in a public issue. Use **Security → Report a vulnerability** in this repository and include the affected version, minimal reproduction, impact, preconditions and suggested mitigation.

请勿在公开 Issue 中提交港口拓扑、设备地址、访问凭证、未公开船舶轨迹、生产遥测或可直接利用的细节。请使用仓库 **Security → Report a vulnerability** 私密报告，并提供受影响版本、最小复现、影响与前置条件以及建议的缓解方式。

We aim to acknowledge a credible report within seven days. Remediation and disclosure timing depend on severity, exploitability and affected operators. Only the latest release is eligible for security fixes; an unreleased branch is not a production support commitment.

目标是在七天内确认可信报告。修复与披露时间取决于严重性、可利用性和受影响运营方。仅最新发布版本接收安全修复；未发布分支不构成生产支持承诺。

## Deployment baseline / 部署基线

- Restrict CORS and place authentication, rate limits and TLS at a reviewed reverse proxy.
- Store TOS, AIS, market and actuator secrets in a secret manager; never commit live values.
- Separate training, evaluation and production identities and storage.
- Keep southbound execution in dry-run with human approval until site safety acceptance is complete.
- Scan uploaded datasets for malware, privacy, licensing and data-poisoning risks.
- Sign and retain dataset/model hashes, evaluation evidence, promotions and rollbacks.
- Run the Linux CI security jobs on the exact release commit before changing repository visibility.

- 限制 CORS，并由经过评审的反向代理提供认证、限流和 TLS；
- 将 TOS、AIS、市场和执行端凭证置于密钥管理器，禁止提交真实值；
- 隔离训练、评测和生产身份及存储；
- 在现场安全验收完成前，南向执行保持 dry-run 和人工审批；
- 对上传数据集进行恶意文件、隐私、许可和数据投毒检查；
- 对数据集/模型哈希、评测证据、晋级与回滚记录进行签名留存；
- 在改变仓库可见性之前，对准确的发布提交运行 Linux CI 安全任务。

This repository is a research and engineering system, not a certified autonomous port controller. / 本仓库是研究与工程系统，不是经认证的港口自主控制器。
