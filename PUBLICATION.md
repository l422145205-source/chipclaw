# 公开版本与复现说明

## A. 立即可运行：工作台回放

Python 3.11–3.13，安装 `requirements-web.txt`，启动 `chipclaw.workbench:app`。这一路径不导入 PyTorch，不需要模型，也不请求远程大模型。28 条记录来自 2026-10-11 实际 `/evidence` HTTP 200 响应。`web/demo/catalog.json` 给出每个原始 JSON 的 SHA-256；数据正文没有改写。

静态托管只支持历史回放。实时检索必须运行 Python 后端，GitHub 仓库页面本身不会提供 GPU 推理。

## B. 公开资源清单

公开代码、训练脚本、冻结训练输入、语料来源指纹、模型选择配置、关键实验摘要和回放记录。大型模型权重、原始厂商 PDF、向量缓存和完整运行日志不放入 Git。

`experiments/` 保存结果摘要；运行时必要配置仍在 `reports/`。报告里的原始 SHA 指向当时输入或权重，用于辨别实验身份，不保证重新下载的厂商现行 PDF 与原版本相同。

## C. 重新训练（需要 CUDA，较耗时）

原实验为 Python 3.13 / PyTorch 2.11 / RTX 5070 Ti Laptop，具体依赖见 `requirements-v4.txt`。CUDA 对应的 PyTorch 安装由使用者按设备选择。克隆仓库不会自动下载权重或重新训练。

首先用 `python scripts/prepare_reproduction.py` 在 `.experiments/reproduction` 创建独立实验副本。该副本保留代码与冻结输入，不复制已经完成的训练结果，避免覆盖原始实验。随后切换到这个副本，按顺序执行：

```bash
python -m pip install -r requirements-v4.txt
python scripts/cache_models_v4.py
python scripts/cache_multilingual_v4.py
python scripts/score_teacher_v3.py
python scripts/train_multilingual_v4.py
python scripts/train_domain_lora_v4.py
```

下载脚本使用已固定 revision，默认通过 hf-mirror.com 拉取公开模型并检查权重哈希。教师分数会重新生成；没有宣称跨硬件逐位确定性。训练脚本检测已有 `protocol.json` 后拒绝覆盖；如需重跑，另建副本。

冻结 `queries.jsonl` 与 `passages.jsonl` 已提供。它们用于受控排序对照，不能当作独立人工问答金标准。完整历史数据构建与审核的上游素材并未全部公开；这条复现路径从冻结训练输入开始。

## D. 实时证据工作台

1. 准备完整 Python 依赖；将新训练的学生与领域 LoRA 权重放在代码约定的 `.cache/v4` 路径，保留对应 `reports/*/selected_model.json`。
2. 执行 `python scripts/prepare_public_corpus.py`：从官方 URL 下载 PDF，逐一检查实验来源 SHA，按冻结的字符区间和清单恢复检索索引。若来源版本变化，脚本明确报错，不将新 PDF 当作旧实验。
3. 执行 `python scripts/extract_v3_geometry.py` 提取固定页面物理行列。视觉路径还需 `encode_visual_v3.py` 及公开视觉模型缓存；仅用多语言学生时不需要视觉向量。
4. PowerShell 设置 `$env:CHIPCLAW_API_WARMUP='none'`，或 bash 设置 `export CHIPCLAW_API_WARMUP=none`。
5. 启动 `python -m uvicorn chipclaw.api:app --host 127.0.0.1 --port 8000`。工作台切换“实时检索”后请求 `/evidence`。

学生重排与领域 LoRA 使用 CUDA。界面的“接口已连接”表示完整 API 已启动，不保证权重和语料已准备好；缺失资源会导致检索失败并显示错误。首次运行包括模型加载，不能和报告的纯重排延迟比较。

原有 `/ask` 生成接口保留，另需 Ollama（默认 qwen3:8b）与对应检索资源。当前工作台聚焦已验证的证据检索 / 声明核验流程，不把无生成的 `/evidence` 结果包装成大模型回答。

## E. 测试

```bash
python -m pip install -r requirements-web.txt pytest httpx
python -m pytest tests/test_workbench.py -q
```

公开 CI 运行无 GPU 的回放和数据契约检查。公开的其余算法测试需要完整科学计算依赖，可运行 `python -m pytest tests -q`；完整历史测试集另有依赖本地原始素材的测试，不全部纳入公开 CI。383 项历史测试属于原本地完整环境，不能当作全新克隆环境的测试结果。
