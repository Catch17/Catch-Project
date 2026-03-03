# 06-LLM 文本情感分类（Text Sentiment Classification）

本项目使用 **DeepSeek LLM API** 完成二分类情感分析（`positive / negative`），并对比两种提示词策略（Prompting）在不同难度数据集上的效果：
- **Zero-shot**：不给示例，直接让模型判断
- **Few-shot**：提供少量示例（in-context examples）再让模型判断

同时实现了评估流水线（evaluation pipeline），自动输出预测结果与指标，便于复现实验与写报告。

---

## 目录结构（Project Structure）

- `src/classify.py`：构建 prompt、调用 LLM、解析输出（parse to `positive/negative/unknown`）
- `src/evaluate.py`：批量推理 + 指标计算（accuracy/precision/recall/F1）+ 输出文件
- `data/samples.csv`：简单集（sanity check）
- `data/hard.csv`：困难集（包含否定/转折/轻微讽刺/更口语的表达）
- `outputs_*/*`：运行后自动生成（predictions + metrics）

---

## 环境准备（Setup）

### 1) 创建并激活虚拟环境（Virtualenv）
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) 安装依赖（Install dependencies）
```powershell
pip install -r requirements.txt
```

### 3) 配置 API Key（Set API key）
```powershell
$env:DEEPSEEK_API_KEY = "YOUR_KEY_HERE"
```

---

## 快速运行（Quick Start）

### Zero-shot
```powershell
python -m src.evaluate --data data/samples.csv --mode zero --outdir outputs_zero
```

### Few-shot
```powershell
python -m src.evaluate --data data/samples.csv --mode few --outdir outputs_few
```

运行结束后会生成：
- `outputs_xxx/predictions.csv`
- `outputs_xxx/metrics.json`

查看指标：
```powershell
type outputs_zero\metrics.json
type outputs_few\metrics.json
```

---

## 数据集说明（Datasets）

### `samples.csv`（easy / sanity check）
文本情绪非常明显，用于确认流程与解析逻辑跑通。

### `hard.csv`（hard set）
刻意加入更接近真实表达的句式：
- 否定（negation）：如 “Not bad.”
- 转折（contrast）：如 “Looks nice, but feels cheap.”
- 轻微讽刺/含糊（mild sarcasm / ambiguity）：如 “It works, I guess.”

目的：更真实地观察 **prompt 策略**对鲁棒性的影响。

---

## 实验结果（Results）

### Hard set：zero-shot vs few-shot

| Setting | Samples | Valid | Unknown | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | 50 | 50 | 0 | 0.9400 | 1.0000 | 0.8500 | 0.9189 |
| few-shot  | 50 | 49 | 1 | 0.9592 | 1.0000 | 0.8947 | 0.9444 |

**结论（Observation）**
- few-shot 在 hard set 上提升了 **Recall 与 F1**，说明给少量示例可以增强模型对否定/转折句式的适应能力。
- 少量 `unknown` 来自模型输出不完全符合要求（没有严格只输出 `positive/negative`）。本项目通过 `parse_prediction()` 将非规范输出归类为 `unknown`，并在 metrics 中统计（更符合工程上的鲁棒性要求）。

---

## 常见问题（Troubleshooting）

### 1) HTTP 402: Insufficient Balance
表示 API 余额不足/未开通计费。解决：
- 去 DeepSeek 控制台充值/开通计费
- 或更换其它提供免费额度的 LLM 平台（后续可扩展）

### 2) 目录找不到 / 没有 outputs
如果运行中途报错，程序会提前退出，可能导致输出目录未生成。请先确保 `python -m src.evaluate ...` 能正常执行完毕。

---

## 可改进方向（Future Work / Improvements）
- 增加更多类别或引入中性（neutral）标签，做三分类
- 加入更系统的错误分析（error analysis）：抽取错分样本，分析失败模式并迭代 prompt
- 增加缓存（cache）/重试（retry）/限流（rate limiting），降低成本并提升稳定性
