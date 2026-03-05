# RiskEval 论文框架复现（arXiv:2601.07767）

这个仓库实现了论文里的三阶段框架：
1. `Solver` 先根据惩罚系数 `lambda` 自行决定 `ANSWER` 或 `ABSTAIN`，并给出置信度。
2. `Parser` 把答案和置信度规范化成结构化值 `p_hat`（0~1）。
3. `Judge` 仅在需要时用于判定答案是否正确（`mcq`/`numeric` 默认走规则判分，不调用 judge 模型）。

当前支持两种 LLM 后端：
- `api`：学校平台 / OpenAI-compatible deployment API
- `local`：本地 Hugging Face checkpoint（`transformers + torch`）

并输出论文核心指标：
- Policy Consistency
- Normalized Regret
- Normalized Utility
- AUARC
- ECE-10
- Brier Score
- Abstention Rate
- Answered Accuracy

## 1) 安装

```bash
cd /Users/wuyuchen/Desktop/RiskEval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果要跑本地 checkpoint，再安装：

```bash
pip install "transformers>=4.41.0" "torch>=2.2.0"
```

## 2) 配置后端（API 或本地）

### 2.1 API Key（学校平台）

```bash
export HKBUEDU_API_KEY="你的key"
python /Users/wuyuchen/Desktop/RiskEval/scripts/test_hkbu_api.py
```

配置里使用：
- `[llm] provider = "api"`

### 2.2 本地 checkpoint（推荐先 smoke 跑通）

配置里使用：
- `[llm] provider = "local"`
- `[local] local_files_only = true`（只用本地已下载权重，不走网络）
- `[models] solver_model/parser_model/judge_model` 填你本地模型 ID（可设为同一个）
- 如需混合后端（例如 solver 本地、parser/judge 用 API），可加：
  - `solver_provider = "local"`
  - `parser_provider = "api"`
  - `judge_provider = "api"`

示例配置：
- `/Users/wuyuchen/Desktop/RiskEval/configs/gpqa_local_qwen25_3b_strategy1.toml`

## 3) 准备数据

默认示例数据在：
- `/Users/wuyuchen/Desktop/RiskEval/data/sample_mcq.jsonl`

每行 JSON 格式：
```json
{"id":"q1","question":"...","choices":["A. ...","B. ...","C. ...","D. ..."],"answer":"C"}
```

当前支持的 `task_type`:
- `mcq`: 多项选择，`answer` 为选项字母
- `numeric`: 数值题，`answer` 为最终数值字符串

如果你已经把 Hugging Face 数据集缓存到了本地，可直接转换：

```bash
python /Users/wuyuchen/Desktop/RiskEval/scripts/prepare_cached_datasets.py --dataset all
```

会生成：
- `/Users/wuyuchen/Desktop/RiskEval/data/gpqa_diamond_test.jsonl`
- `/Users/wuyuchen/Desktop/RiskEval/data/gsm8k_main_test.jsonl`
- `/Users/wuyuchen/Desktop/RiskEval/data/gsm8k_socratic_test.jsonl`
- `/Users/wuyuchen/Desktop/RiskEval/data/hle_test.jsonl`
- `/Users/wuyuchen/Desktop/RiskEval/data/hle_mcq_text_only_test.jsonl`

说明：
- `GPQA`: 直接按 `mcq` 处理
- `GSM8K`: 按 `numeric` 处理，并从 `#### final_answer` 提取 gold
- `HLE`: 会导出全量 `hle_test.jsonl`；另外 `hle_mcq_text_only_test.jsonl` 是 `multipleChoice` 且无图片的 text-only 子集

## 4) 运行复现

```bash
riskeval --config /Users/wuyuchen/Desktop/RiskEval/configs/example.toml
```

如果是 API 后端，当前 HKBU 平台按 deployment-style chat completions 调用，配置里需要：
- `base_url = "https://genai.hkbu.edu.hk/api/v0/rest"`
- `api_version = "2024-12-01-preview"`

如果是本地后端，重点配置项：
- `[local].device = "auto"`：会优先用 `mps`（Apple Silicon）
- `[local].dtype = "auto"`：`mps/cuda` 默认 `float16`
- `[local].local_files_only = true`：确保不会消耗线上 quota

## 5) 输出文件

在 `out_dir`（默认 `/Users/wuyuchen/Desktop/RiskEval/output/run1`）下：
- `example_runs.jsonl`：逐样本逐 penalty 明细
- `example_runs.csv`：同上，CSV 版
- `summary.json`：按 penalty 聚合后的指标

## 6) 画图与表格

运行结束后，可从一个或多个实验目录画图：

```bash
python /Users/wuyuchen/Desktop/RiskEval/scripts/plot_riskeval_results.py \
  --dataset-name "GPQA Diamond" \
  --output-dir /Users/wuyuchen/Desktop/RiskEval/output/plots/gpqa \
  --run "GPT-5-mini=/Users/wuyuchen/Desktop/RiskEval/output/gpqa_gpt5mini_strategy1"
```

会生成：
- `main_metrics.png`：Average Confidence / Normalized Average Utility / Policy Consistency
- `calibration_metrics.png`：AUARC / ECE-10 / Brier
- `abstention_by_penalty.png`：不同 penalty 的 abstain 数量柱状图
- `answered_count_by_confidence_bin.png`：按置信度分箱的回答数量柱状图
- `abstention_rate_by_penalty.png`：不同 penalty 的 abstain rate 折线图
- `answered_accuracy_by_confidence_bin.png`：按置信度分箱的回答准确率折线图
- `analysis_dashboard.png`：4 图总览（abstentions / abstention rate / answered count / answered accuracy）
- `metrics_table.csv` 与 `metrics_table.json`：高 penalty 区间（`lambda >= 10`）的汇总表

## 7) Prompt 策略

在配置里改 `run.prompt_strategy`：
- `1`: baseline confidence prompt
- `2`: counterfactual confidence prompt（论文对比策略）

## 8) 注意

- 本实现遵循论文公开描述（含附录 prompt 思路）构建，可直接用于你们平台 API。
- 如果你有论文原始数据集（如 MMLU-Pro / GPQA / MATH-500 / GSM8K / MMLU），把 `data_path` 改成对应 JSONL 即可跑完整实验。
