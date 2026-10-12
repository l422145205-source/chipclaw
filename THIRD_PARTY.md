# 第三方来源与归属

项目代码与模型、厂商手册、派生片段应分别理解。此仓库未替第三方材料授予新的许可。

- 芯片手册来自 STMicroelectronics、Raspberry Pi、Texas Instruments、Bosch Sensortec；官方 URL 和来源 SHA 见 `data/corpus-v3/index/manifest.json`。回放中的摘录保留来源链接、页码和指纹；原始手册版权属于相应权利人。
- 多语言排序基座：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，固定 revision 见实验 protocol。
- 领域排序基座：`Qwen/Qwen3-Reranker-0.6B`，固定 revision 见实验 protocol。
- 视觉检索：`vidore/colSmol-256M` 与其固定版本基座，见 `reports/v3-start/model_cache.json`。
- 历史 E5 / 英文 MiniLM 路径用于先前对照；模型身份见 `chipclaw/config.py`。

使用或分发第三方模型和派生权重前，应分别阅读对应模型卡的许可。当前公开仓库不附带大体积预训练或微调权重。
