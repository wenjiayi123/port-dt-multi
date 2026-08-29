# 国际海事标准互操作与符合性证据

本模块把项目内部靠泊事件、海事申报语义和港区水文地理产品目录转换为可复核、可追溯、失败关闭的映射证据。它解决的是“同一船舶、同一港口、同一靠泊过程能否在不同标准语义之间一致表达”，不负责向主管机关报送，也不是电子海图显示与信息系统。

## 固定标准基线

截至二〇二六年八月二十六日，合同固定为：

- 数字化集装箱航运协会港口靠泊标准第二版：映射港口靠泊、码头靠泊、港口服务、船舶、预计／请求／计划／实际时间戳和装卸量预报，并覆盖取数与推送两类时间戳、装卸量预报符合性场景。
- 国际海事组织便利运输通函第五十六号现行数据目录：当前只实现一般申报到港或离港最小语义子集，固定核对船舶国际海事组织编号、船名、港口联合国口岸及相关地点代码、预计到达或离开时间。
- 国际水道测量组织通用水文数据模型第五点二点一版，以及港区相关产品：电子航海图第二版、水深表面第三版、水位信息第二版、表层流第二版、航行警告第二版、富余水深管理第二版。

权威资料入口：

- [数字化集装箱航运协会港口靠泊标准实施指南](https://reference.dcsa.org/content/standards/releases/port-call/v2-0-0/port-call-v2-0-0-implementation-guide)
- [国际海事组织海事单一窗口与现行数据目录](https://www.imo.org/en/ourwork/facilitation/pages/maritimesinglewindow-default.aspx)
- [国际水道测量组织现行电子海图与通用水文数据模型标准](https://iho.int/en/enc-ecdis)

标准升级必须修改服务中的固定版本、测试样例和现场测试报告，不能只改页面文字。

## 三层证据

一、内部映射合同：

- 靠泊事件和六方协同证据摘要必须为小写安全散列摘要；
- 港口靠泊、码头靠泊、服务和事件标识采用规范通用唯一标识；
- 船舶国际海事组织编号必须通过校验位核验，港口采用联合国口岸及相关地点代码；
- 至少覆盖一条服务时间戳和一条装卸量预报；
- 海事单一窗口身份、港口和时间必须与同一靠泊对象一致；
- 六类水文产品必须使用固定生效版本，目录必须带生产者、时效、覆盖范围和数据摘要，覆盖范围必须包含港口坐标。

二、外部符合性与现场受理测试：

- 数字化集装箱航运协会报告必须覆盖取数和推送的时间戳、装卸量预报四类固定场景；
- 海事单一窗口报告必须来自授权测试环境，并验证一般申报最小语义子集；
- 通用水文数据模型报告必须覆盖六类固定产品；
- 三份报告都必须绑定本次 `mapping_digest`。事件、时间或产品发生变化后，旧报告会被拒绝。

三、现场独立复核：

- 来源为 `authorized_site_interoperability_export`；
- 水文目录标明官方或授权生产者来源；
- 数据治理、海事主管责任方和水文资料责任方由三名互异且独立于数据所有者的复核人确认；
- 变更工单、来源核验和三类外部报告同时存在。

只有三层证据全部通过，`site_interoperability_accepted` 才能为真。即使通过，`authority_submission_allowed`、`navigational_use_allowed`、`official_certification_claim_allowed`、`dispatch_allowed` 和 `production_authority` 仍固定为假。

## 版本化核验

内部合同样例：

```bash
.venv312/bin/python -m scripts.verify_maritime_interoperability \
  --input private/interoperability_input.json \
  --output private/interoperability_evidence_v1.json
```

现场证据还需要显式提供来源证明、三方复核和变更工单：

```bash
.venv312/bin/python -m scripts.verify_maritime_interoperability \
  --input private/interoperability_input.json \
  --output private/interoperability_evidence_v2.json \
  --authorized-source-attested \
  --data-governance-approved-by reviewer-data \
  --maritime-authority-approved-by reviewer-maritime \
  --hydrographic-authority-approved-by reviewer-hydrographic \
  --change-ticket change-2026-001
```

命令拒绝覆盖已有文件。生产环境通过 `PORT_DT_MARITIME_INTEROPERABILITY_PATH` 指向新生成的私有证据；该证据必须与孪生图谱、现场标定、影子验收、执行联锁和靠泊协同证据使用同一个 `site_id`。

## 页面演示边界

“标准互操作”页签的“运行映射核验”只把浏览器内固定合同样例发送给无副作用映射接口：

- 不保存现场数据；
- 不调用外部符合性沙箱；
- 不连接海事单一窗口；
- 不加载或显示适航电子海图；
- 不发送设备或调度指令。

页面结果中的“合同通过≠外部符合性”是固定业务边界，不是临时提示。
