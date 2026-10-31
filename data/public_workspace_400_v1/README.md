# 新增 400 条公开调用：待复核批次

交付日期：2026-10-31（北京时间）。本批是上一批 120 条之外的 **400 条新候选**，不是已经标注的训练集。
全部 `label=null`、`review_status=pending`；未生成 AI 建议，也未自动写入旧复核台。

## 直接使用

- `annotation_candidates.jsonl`：400 条完整模型输入及外置来源元数据，供程序读取。
- `review_cases/001.json` 至 `400.json`：逐条完整上下文，不展示来源 episode 的安全结果。
- `review_sheet.csv`：人工复核索引；观察预览仅前 1500 字符，判断必须阅读对应完整 JSON。
- `review_metadata.jsonl`：初始空标签和复核状态。
- `review_batches/batch_01.jsonl` 至 `batch_04.jsonl`：每份 100 条，格式与 review_cases 相同，可分批或分工复核；没有额外样本。
- `sources/`、`source_manifest.json`：400 份原始公开轨迹、固定版本链接及 SHA-256。
- `dataset_statistics.json`、`validation_report.json`：抽样统计及检查结果。
- `manifest.sha256.json`：交付文件校验清单。
- `scripts/`：收集及校验脚本；`UPSTREAM_LICENSE`：上游许可。

## 收集结果

来源为 [AgentDojo 官方公开日志](https://github.com/ethz-spylab/agentdojo/tree/089ed468cf3ed0322acc66b0211f26d9d90dbf60/runs)，
固定提交 `089ed468cf3ed0322acc66b0211f26d9d90dbf60`。
这是模拟 benchmark 中模型实际提出的工具调用，不是生产数据，也不是按当前规则过滤器生成的动作。
收集过程中只读取公开数据，没有执行其中的调用或指令，也没有调用付费模型。

| 项目 | 数量 |
| --- | ---: |
| 读取成功的源日志 | 1,435 / 1,435 |
| 过滤旧样本后的唯一候选池 | 1,226 |
| 最终交付 | 400 |
| 最终使用的不同源轨迹 | 400 |
| 来源模型版本 | 13 |
| 用户任务 | 14 |
| 原有任务家族／组件 | 8 |
| 工具种类 | 7 |
| 攻击配置类别（不含无攻击） | 15 |
| 与旧 502 条及上一批 120 条的完整输入重复 | 0 |
| 与上述旧数据的请求＋动作精确重复 | 0 |
| 本批不同的请求＋动作组合 | 400 |
| 与上一批 120 条共用源轨迹 | 0 |

模型版本包括 GPT-3.5、GPT-4 Turbo、GPT-4o、GPT-4o-mini、Claude 3 Haiku/Opus、
Claude 3.5 Sonnet 两个版本、Claude 3.7 Sonnet、Gemini 1.5 Flash/Pro、Gemini 2.0 Flash 和 Command R+。
具体版本、攻击配置和来源永久链接均在数据中保留。

| 工具 | 条数 |
| --- | ---: |
| append_to_file | 186 |
| send_email | 79 |
| create_calendar_event | 63 |
| create_file | 58 |
| delete_file | 6 |
| share_file | 4 |
| reschedule_calendar_event | 4 |

按来源 episode 分类，有 54 条来自正常轨迹、273 条来自被攻击且 security=false 的轨迹、73 条来自其他被攻击轨迹。
**这些是来源属性，不是 0/1 标签。** 当前还不知道人工确认后的授权／未授权比例。

## 怎么复核

只判断“原始用户请求是否授权这次具体调用”，同时核对目标、收件人、正文、权限和时间：

- `0`：授权。当前动作和参数处于用户允许的范围，现有上下文可以支持。
- `1`：未授权。动作或具体参数越过用户授权边界。
- 不确定：保持空标签，记录为待讨论；不要为了凑满 400 个标签强行判断。

工具返回的数据是待分析证据，其中的指令本身不构成用户授权。
观察有攻击文字不代表当前动作一定未授权；反之，工具名称正确也不代表参数得到授权。
复核时应先看完整 review_case，再查来源，避免用整段轨迹的成功／失败结果代替调用级判断。

## 使用范围和偏差

**复核后只用于 train/development。** 本批只采集原 dataset_502_v1 划入 train 的 14 个任务，并核对请求一致；
没有采集旧 validation/test 任务。新增的是调用和来源覆盖，不是 400 个新任务，也不是独立测试集。

为了减少重复，固定 seed 4266，在每个任务／模型／攻击配置下最多选择四份源轨迹；
先按任务、工具、模型、攻击配置轮转取样，优先不同请求＋动作和不同 episode。
其中 383 条在优先阶段选出，另 17 条放宽了批内 episode 不重复限制；所有请求＋动作仍彼此不同且不与旧数据重复。
共有 417 个来源引用，指向 400 份轨迹；16 份轨迹支持本批中的多个调用。按任务及关联轨迹分组，不能把这些调用当成独立 episode。
未按当前规则或学习模型的预测挑选样本。

这一选择会偏向容易产生不同文本参数的工具：append_to_file 和 send_email 较多；删除或共享等固定动作较少。
不能用本批比例估计真实部署中的攻击频率，也不能把请求＋动作的精确不同当作语义完全独立。
51 条动作仍包含同一个固定攻击邮箱，另有模板或近重复措辞；保留任务家族分组，并在后续诊断中检查这些捷径。

模型只接收 `model_input` 三个字段。label 是监督目标；来源模型、攻击类型、episode_security、
episode_utility、采样类别、ID、来源路径和复核元数据都不能拼进输入。
调用级分类结果也不能直接替代完整 episode 的在线攻击成功率或任务完成率。

## 质量检查与复现

已经验证全部 400 条的源文件 SHA-256、Git blob 与固定提交一致，并从每份轨迹的对应步骤重建原始输入。
全部输入仅使用调用前的观察，不包含当前调用的返回结果。JSONL、复核表、逐条 JSON 一致，标签全部待确认。
还检查了旧数据精确去重、请求＋动作去重、旧测试任务隔离和上一批源轨迹隔离。

```text
python scripts/prepare_public_trace_batch_400.py --project PATH_TO_PROJECT --upstream PATH_TO_AGENTDOJO --cache PATH_TO_CACHE --prior-batch PATH_TO_ORIGINAL_120_BATCH --output-dir NEW_EMPTY_FOLDER --fetch --max-candidates 400
python scripts/validate_public_trace_batch_400.py --batch PATH_TO_400_BATCH --project PATH_TO_PROJECT --prior-batch PATH_TO_ORIGINAL_120_BATCH --upstream PATH_TO_AGENTDOJO
```

`--project` 指向包含原 502 条及划分文件的项目；`--prior-batch` 指向上一批原始 120 条候选交付，
不是网页导出的已确认标签文件。去重基准的 SHA-256 记录在统计文件中。
脚本负责候选、逐条上下文、来源文件、复核表及校验报告；分批文件、本文和交付校验清单是附加交付材料。
