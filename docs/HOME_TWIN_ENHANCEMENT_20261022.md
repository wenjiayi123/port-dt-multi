# 电脑端港区孪生首页增强

本轮保留原有导航、业务模块及生产权限边界，扩展三维场景和可用服务信息。

## 对标参考

- [新加坡 MPA Maritime Digital Twin](https://www.mpa.gov.sg/maritime-singapore/innovation-and-r-d/maritime-digital-twin)：同一视图中的船舶、天气、运行活动，历史回放及情景分析。
- [鹿特丹港 Smart infrastructure](https://www.portofrotterdam.com/en/port-future/smart-infrastructure)：基础设施、船舶活动、气象与水文信息。
- [鹿特丹港 Weather, tides and water depth](https://www.portofrotterdam.com/en/up-to-date-information/weather-tides-and-water-depth)：环境数据、潮位与航运服务信息的组织方式。

以上用于能力和交互设计参考，不表示本项目已具备同等级的现场数据、测绘精度、正式认证或运行部署。

## 已实现

| 能力 | 本轮实现 | 数据性质 |
|---|---|---|
| 港区设施 | 编号岸桥/堆场、波纹箱体、船体/栏杆/舷窗、护舷/系缆设施、车道、岸电储能、光伏顶棚、闸口和铁路 | PortViz 几何加工程示意设施；非上海港实测 BIM |
| 动态表达 | 水面、车辆插值、岸桥小车与吊具联动、繁忙状态、日夜灯光 | 帧驱动投影及明确的示意动画；非实测设备运动学 |
| 空间操作 | 六种视角、巡航、四种图层、标签开关、小地图、对象详情、全屏 | 使用同一场景和现有路由，无业务副本 |
| 历史查看 | 最近 120 个已接收帧，拖动即暂停，返回最新帧继续 | 仅本次会话缓存，不宣称后端任意历史时刻检索 |
| 气象水文 | 风、潮、浪、气温与 24 点曲线 | 现有上海公开校准/再分析序列，显示来源时间；与 PortViz 来源独立 |
| 靠泊服务 | 到港、引航、拖轮、系泊、装卸、离港服务链和接入状态 | 未配置现场源时显示待接入，不造船名、IMO、ETA |
| 能源服务 | 岸电、储能、冷藏箱当前负荷及设备曲线联动 | 现有遥测契约的校准回放；不推断现场 SOC 或接电完成 |
| 协同服务 | 11 类资源、9 阶段、规划步长与冻结窗口 | 现有只读资源合同；现场证据另行核验 |

## 接口与实现

首页使用现有 `/api/portviz/bootstrap`、`/api/portviz/stream`、`/api/assets`、`/api/telemetry/recent/{asset_id}`、`/api/v3/external-signals/evidence`、`/api/v3/port-call/readiness`、`/api/v3/end-to-end-coordination/readiness` 等只读接口。本轮没有新增生产控制或计划写入接口。

容器使用实例化绘制；帧缓存上限 120；标签仅在投影变化时更新 DOM；切换模块、隐藏页面或打开服务详情时停止三维连续绘制，返回后恢复。三维资源仍使用本地 Three.js，保留加载失败提示与重试。

要进一步成为指定码头的高保真现场孪生，需要经授权的测绘/BIM、设备尺寸与坐标、现场船期/AIS、遥测、事件及互锁回执；本轮不将工程示意包装成上述现场能力。
