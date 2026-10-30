# 角色 C 开发集诊断与交接

本次基于远程 `lxy/offline-model-training` 的提交 `beed1abc7b5448d6bdb084dd4d9fca747166dfe4` 工作。没有运行 AgentDojo 或调用 DeepSeek。以下新实验只使用 train 和 validation；本次未读取或评价 test。标签仍沿用原始 AI 辅助初标，不能视为已复核的 gold labels。

已提交的[结果与逐条预测](diagnostics/lzs-v1/README.md)和[待复核配对样本](../data/diagnostics/lzs_pairs_v1/README.md)可直接在仓库查看。依赖快照见 [requirements](../configs/lzs_diagnostics_requirements.lock.txt)。Prompt auditor 的后续实现另见[交接说明](prompt_auditor_handoff.md)。

## 主要结论

当前优先事项是数据对照和输入证据检查。仅调整模型复杂度没有解决来源与标签完全重合的问题。这次得到的关键证据是：TF-IDF 只看 proposed action，验证集仍达到 F1=1.000。该动作中含有真实收件人、参与人和权限等内容；结果提示数据的动作表面模式已足以区分当前标签，不证明模型理解了动作与用户授权的关系。

## 实际运行结果

全部 TF-IDF 实验沿用队友的词/字符 n-gram、balanced Logistic Regression 和 seed 4266。分别只在 train 拟合向量化器和分类器，各自在 validation 按原策略选阈值：recall 至少 0.8，然后最小化误拦率。validation 是开发和选阈值数据，不是独立最终评估。

| 输入 | validation 正类 F1 | recall | 授权误拦率 | Average Precision | 阈值 |
|---|---:|---:|---:|---:|---:|
| 完整输入 | 0.576 | 0.810 | 0.404 | 0.670 | 0.48 |
| 移除用户请求 | 0.600 | 0.857 | 0.404 | 0.659 | 0.50 |
| 移除 observations | 1.000 | 1.000 | 0.000 | 1.000 | 0.47 |
| 仅保留动作 | 1.000 | 1.000 | 0.000 | 1.000 | 0.59 |

原代码字段名叫 `pr_auc`，实际调用 scikit-learn 的 average_precision_score；本表准确称作 Average Precision，不能当成额外计算的梯形 PR 曲线面积。

action-only 模型的较大正向权重包含特定邮箱的人名/编号片段和 `permission rw`；负向权重包含特定企业邮箱域和文件 ID。这些系数支持模型利用具体实体或权限模式的解释，但单个系数不能证明某条预测的因果来源。详细权重保存在 action_only_top_features.json。

完整输入下 validation 的 21 个误拦全部来自 create_calendar_event：其 39 条授权调用中误拦 21 条。按任务看，user_task_9 的 13 条授权样本误拦 12 条，user_task_21 的 13 条误拦 9 条；user_task_12 与 user_task_34 的授权样本未误拦。不要把包含多条相关样本的任务当成同数量的独立证据。

阈值诊断也说明不能只靠改阈值同时解决两端表现。在 validation 的观测分数候选上，FBR<=5% 时最高 recall 为 9/21=42.9%（误拦 2/52）；FBR<=10% 时为 11/21=52.4%（误拦 4/52）。这是同一验证集上的描述性曲线，没有据此替换既有阈值策略或评价 test。

## Transformer 输入审计

使用公开 DistilRoBERTa tokenizer.json，快照 revision 为 `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`，SHA-256 为 `847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e`。直接使用该文件的 tokenizers backend；不是 lxy 原运行的 tokenizer 工件，不保证与未记录版本的原运行完全一致。

按原代码 96 request / 256 observation / 156 action 的预算：

| 项目 | Train 341 条 | Validation 73 条 |
|---|---:|---:|
| request 超预算 | 0 | 0 |
| 选中 observation 文本超预算 | 336 | 72 |
| action 超预算 | 116 | 0 |
| 因只选最多三条而丢弃部分 observations | 119 | 0 |
| 解码 action 中缺少列出的关键字符串 | 0 | 0 |

审计覆盖收件人、参与人、file_id、附件等代码列出的字符串。没有发现这些字符串丢失，修正了先前“目标参数可能被截掉”的未验证猜测；仍不能据此确认授权证据完整。body、时间及其他未列入关键字段的内容可能被截断，observation 中间的授权依据也可能丢失。

因此下一步应人工检查开发集错误对应的完整观察与实际编码内容，而不是直接增大输入长度。不要只保留注入片段；正常授权证据同样需要保留。任何预处理修改都需重新在 train/validation 检查，不能按 test 特例修补。

## 配对探针草稿

生成 20 条、10 对 synthetic development probes，来自两类模板的地址变体：邮件收件人和日程参与人。它们不是十个独立 benchmark 任务，也不是正式 challenge benchmark。

每对 request/observations 相同，只替换一个 recipient/participant。用户明确只允许 Owner，观察中 malicious block 复现 Owner 邮箱，却通过 External Reviewer 别名指向越权目标。这样可以在同一上下文中检查授权关系与词面复制是否被混淆。

候选 `label` 全部 null；预期 0/1 及理由单独放在 review_metadata.jsonl 和 review_sheet.csv，所有 review_status=pending。需要真人看完整上下文，只有复核后才可建立正式标签。该文件不应混进现有 502 行训练集。

规则阈值固定为从原 validation 得到的 0.65。在这份刻意设计的草稿上，相对预期标签的结果：10 条授权全部放行，10 条越权也全部放行。TF-IDF 完整输入模型则放行所有邮件并阻断所有日程，得到 5 个误拦和 5 个漏拦。两者准确率都为 50%，但失败机制不同。

这些结果是开发探针上的预期标签比较，未经独立审核；不能替代真实攻击 ASR、泛化指标或最终论文主表。它们的价值是提供可以由队友核对的具体反例。

## 已补的代码与限制

- 两个训练入口都新增 full / no_request / no_observations / action_only 模式。
- 新增 development-only 选项，仅打开 train/validation。默认行为仍兼容原有全划分运行。
- checkpoint 补记 pooling 和 ablation，防止推理时混淆。
- 输出目录非空时拒绝写入，避免不同实验混在一起。
- 包初始化改为按需加载 Gate，使离线建模不用导入整个 AgentDojo 依赖栈。
- 新增自动化开发诊断、输入审计和配对探针生成脚本。

本机系统 Python 的 numpy/scikit-learn 存在二进制兼容问题，本次使用隔离环境，未更改系统环境。实际为 Python 3.12，numpy 2.5.4、scikit-learn 1.9.1、joblib 1.6.0；精确依赖快照随交付提供。原运行没有锁版本，因此这是当前代码在记录环境下的复现，不是逐位复现队友运行。

建模相关原测试、新增消融及开发模式测试、split 测试共 13 项通过。开发模式的集成测试在没有 test 文件的临时目录成功运行。训练入口和新增脚本通过编译与 diff whitespace 检查。没有重跑 AgentDojo 全套测试；没有训练 Transformer，也没有原 Transformer checkpoint 可供逐例错误复核。Transformer 新增选项只有代码和编译检查，实际训练验证仍待有完整环境时完成。

## 如何在队友分支复用

先切到 lxy/offline-model-training 的上述基准提交或确认后续版本兼容。在仓库根目录覆盖代码包中的对应文件；先审阅 git diff，不要同时覆盖文件又应用同一 patch。包内补丁也可用 `git apply --check` 先检查，再用 `git apply`。

PowerShell 示例（使用你们已有可用建模环境）：

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
python -m pytest -q tests/test_modeling.py tests/test_development_diagnostics.py tests/test_dataset_split.py
python scripts/build_diagnostic_pairs.py --output-dir results/lzs_probes_new
python scripts/run_development_diagnostics.py --output-dir results/lzs_diagnostics_new --probe-dir results/lzs_probes_new
```

以上诊断 runner 会跑四个 TF-IDF 版本并输出 validation 分析，不会运行 Transformer 或调用任何 LLM API。输出目录每次选新名称。

主模型消融入口已经准备好，有可用 torch/transformers 环境时可分别运行：

```powershell
python scripts/train_frozen_transformer.py --development-only --ablation full --output-dir results/lzs_transformer_full
python scripts/train_frozen_transformer.py --development-only --ablation no_request --output-dir results/lzs_transformer_no_request
python scripts/train_frozen_transformer.py --development-only --ablation no_observations --output-dir results/lzs_transformer_no_observations
```

输入审计需要 `tokenizers` 和 tokenizer.json。可在有网络时下载官方分词器，记录快照与哈希，再运行：

```powershell
python scripts/audit_verifier_inputs.py --tokenizer-json PATH_TO_TOKENIZER_JSON --output-dir results/lzs_input_audit_new
```

## 现在分给你的具体工作

1. 把 20 条草稿交给 B/A/D 复核：重点确认“动作与用户授权是否一致”，不要先看模型预测；本次没有代你发消息。
2. 与 A 确认能否从真实攻击轨迹找到 observed unauthorized。没有就如实保留限制，不把 constructed 称为真实攻击。
3. 为 B 准备与构造正例同处理方式的授权对照，优先在同一请求与上下文中最小改变目标/权限。
4. 与 lxy 核查开发集两类高误拦任务的输入证据，随后再跑主模型的两个消融，保持原 test 封存。
5. Prompt-only auditor 和在线 verifier 接口仍是后续可由你承担的模块；本次交付聚焦瓶颈诊断，没有声称已完成它们。

若修订数据后继续开发，现有 test 已被团队看过，需准确披露使用过程。需要更强最终证据时，应预先锁定新的未参与开发的任务/攻击评估集合，不能仅重新随机切同一批样本来恢复独立性。
