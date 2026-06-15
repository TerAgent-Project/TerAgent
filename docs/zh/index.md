# TerAgent 文档

欢迎使用 TerAgent 文档。TerAgent 是一个基于编译器-适配器架构构建生产级 AI Agent 系统的 Python 库。

## 指南

| 指南 | 说明 |
|------|------|
| [快速上手](getting-started.md) | 安装、快速入门和第一步 |
| [架构](architecture.md) | 设计原则、模块依赖、数据流 |
| [安全](security.md) | 权限系统、沙箱、2PC 文件写入、API Key 安全、跨平台兼容性 |
| [配置](configuration.md) | agent.toml、类型化配置、环境变量、平台专属路径 |
| [流式执行](streaming.md) | 流式工具执行、调度策略、降级 |
| [自强化学习数据](self-rl.md) | TAP 追踪、DPO 对生成、数据宪章 |
| [贡献指南](contributing.md) | 开发环境搭建、编码规范、添加模块 |
| [四模型适配指南](adaptation_guide.md) | DeepSeek V4、MiniMax M3、GLM-5、GLM-5.2 配置与最佳实践 |
| [GLM-5.2 使用指南](../en/glm_52_guide.md)（英文） | 1M 上下文、双思考模式、PreservedThinking、5V-Turbo 视觉协调 |
| [长时任务指南](../en/long_horizon_guide.md)（英文） | GLM-5/5.2 8小时+自主任务 |
| [多模态指南](../en/multimodal_guide.md)（英文） | MiniMax M3 图像、视频和桌面操作 |

## 报告与部署

- [四模型评估报告](../EVALUATION_FOUR_MODELS.md) — DeepSeek V4、MiniMax M3、GLM-5、GLM-5.2 评估结果
- [GLM-5.2 稳定性报告](../glm_52_stability_report.md) — GLM-5.2 生产稳定性验证
- [昇腾部署指南](../deployment_guide_ascend.md) — 在华为昇腾 NPU 上部署 TerAgent

## API 参考

- [完整 API 参考](api-reference.md) — 逐模块参考及代码示例

## 快速链接

- **项目**: [GitHub](https://github.com/teragent/teragent)
- **许可证**: Apache License Version 2.0
- **版本**: 0.1.2
