# AgentDojo Workspace 120：LZS 复核结果

LZS 已完成 120 条调用的逐条复核：**86 条授权（0），34 条未授权（1），0 条待讨论**。
复核界面显示 AI 建议，因此本批属于 **单人人工、AI 辅助复核**，不是独立盲审或多人共识标签。
采用仅选择标签的复核方式，空的 reason/evidence 字段是预期格式，不代表尚未确认。
结构校验通过不等于逐条语义标签已被第二人验证。

## 文件与读取

| 文件 | 用途 |
| --- | --- |
| `workspace120-lzs-2026-10-30-reviewed.jsonl` | 120 条已确认标签和完整原始 model_input，模型侧读取入口 |
| `workspace120-lzs-2026-10-30-backup.json` | 原始复核台备份，包含时间、状态、复核人、AI 辅助记录 |
| `workspace120-lzs-2026-10-30-review.csv` | 人类可读复核清单 |
| `provenance.jsonl` | 按 candidate_id 关联原始任务家族、来源轨迹、步骤和永久链接；不能作为模型特征 |
| `validation_report.json` | 数量、标签分布、原始导出文件哈希及交付检查结果 |
| `UPSTREAM_LICENSE` | AgentDojo 来源许可 |

三个用户导出文件按原始字节保存。文件名日期来自网页的 UTC 日期；备份导出时间
`2026-10-30T18:45:15.795Z` 对应北京时间 **2026-10-31 02:45:15.795**。

现有模型读取器支持该 JSONL。从仓库根目录、已安装项目依赖的环境执行：

```python
from tool_call_auditor.modeling.data import load_split

rows = load_split(
    "data/annotations/agentdojo_workspace_120_v1/"
    "workspace120-lzs-2026-10-30-reviewed.jsonl"
)
assert len(rows) == 120
```

模型只读取 `model_input` 中的原始请求、调用前观察及拟执行动作；`label` 是监督目标。
不要将来源模型、攻击类型、安全结果、样本 ID、复核信息或 task_group_id 拼进模型输入。
本目录未自动合并进旧 train 文件，也没有重划旧 validation/test。

## 推荐给模型组的下一步

1. 在这批数据上运行现有固定规则和已有模型，保存逐条预测，分析误拦与漏检。这是开发诊断，不是独立测试成绩。
2. 对有争议的样本及抽样样本做第二人交叉检查，保留标签修订记录。
3. 如用于扩充训练，按 `candidate_id` 关联 provenance，保留原任务家族归属、来源和输入去重信息；只能加入 train。训练前固定实验方案，阈值仍在原 validation 上选择。
4. 保持原 test 隔离；若需评估新攻击来源的泛化，另行收集并预先锁定独立评估数据。本批上的反复诊断和调参不能计作最终测试。

## 来源、划分与限制

数据来自 [AgentDojo 官方公开运行日志](https://github.com/ethz-spylab/agentdojo/tree/089ed468cf3ed0322acc66b0211f26d9d90dbf60/runs)，
固定提交 `089ed468cf3ed0322acc66b0211f26d9d90dbf60`。这是模拟 benchmark 内模型实际提出的调用，不是生产用户数据。
每条的源文件、SHA-256、调用位置和具体永久链接见 `provenance.jsonl`。

- 120 条来自 110 份来源轨迹，覆盖 14 个用户任务、8 个已声明任务家族／组件和 8 种工具。
- 只采集原 dataset_502_v1 分到 train 的任务，复核后仍限定 **train/development**。不能将这批随机拆成一个声称独立的新 test。
- 收集时与旧 502 行完全相同的调用输入已排除；21 条与旧数据的 request+action 相同，但调用前观察不同。
- 18 条动作包含同一固定攻击邮箱，仍存在模板和目标捷径，不能把“公开日志来源”当作消除了所有偏差。
- 收集时正常 episode、被攻击且 security=false 的 episode、其他被攻击 episode 各抽 40 条。来源分层不是调用标签；当前 86/34 来自 LZS 的调用级确认。
- 这些调用级标签不能直接计算在线攻击成功率或证明 AgentDojo 防御效果；在线指标需要完整 episode 评估。

## 交付检查

三个导出的 120 个唯一 ID、标签、状态和复核信息相互一致；全部 model_input 与收集时原始候选一致；
全部任务仍属于原 train 范围。详见 `validation_report.json`。导出格式可由项目现有读取器加载。
