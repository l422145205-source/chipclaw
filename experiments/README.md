# 实验依据索引

- `multilingual-v4/`：训练协议、无 KL / KL 对照、基座留出对照与 tokenizer 审计。
- `domain-lora-v4/`：领域 LoRA 协议、逐 epoch dev 历史及复用留出摘要。
- `adaptive-domain-v4/`：真实候选测试与门控校准结果；门控关闭。
- `visual-v3/`：240 页固定视觉子集上的文本 / 视觉 / 融合对照。
- `runtime-v4/`：真实 HTTP 探针与 20 手册冒烟检查。
- `packets-v4/`：物理证据包和预算闭包检查。

这些是冻结本地实验的结果摘要，不是 GitHub CI 重新运行的模型结果。完整回放响应见 `web/demo/`，训练输入见 `data/ranking-v3-frozen/`。时间指标取决于原实验设备与计时边界。
