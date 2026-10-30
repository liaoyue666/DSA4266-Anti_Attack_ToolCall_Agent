# Prompt-only auditor 交接说明

本次补齐角色 C 的 prompt-only DeepSeek comparator。代码基于 lxy/offline-model-training 的 beed1ab 和上一轮诊断修改；没有执行工具动作、没有调用付费模型，也没有生成任何真实 auditor 准确率。

## 已完成并实际检查的内容

新增 PromptAuditor 模块与 run_prompt_auditor 入口，支持 full / no_request / no_observations / action_only。默认仅生成请求预览。入口只允许 train/validation，不提供 test 开关。

对真实 validation 的 73 条样本生成了请求队列，每条保存 request key；真正发给模型的 payload 只含请求、选中并截断的 observations、动作以及固定审核 prompt。candidate_id、label 和审核/构造 metadata 都在 payload 外。

本次建模、消融、split 与 auditor 相关测试共 29 项通过。测试使用模拟 provider 响应，覆盖 JSON 校验、重复字段、截断或空响应、错误分母、缓存、调用预算及无 API key 的 dry-run；不代表模型判断正确率。代码通过编译与 diff whitespace 检查。

## 输入与实验控制

和 TF-IDF 一样，最多选择 3 条 observations，含最近的注入线索 observation，并将选中文本按头尾保留规则裁到 8000 字符；request 与 proposed action 不裁剪。为了保留字段边界，外层使用结构化 JSON，而非要求模型从一大段文本中自行推断字段。嵌入 observation 的假标题或角色声明仍被定义为不可信数据。

这与 Frozen Transformer 的 512-token 输入不完全相同。比较时应准确写成“相同来源字段与相同 observation 选择，长度预算不同”；不能声称所有方法看到了完全相同的 token。如果要做严格同输入预算对比，需另注册相同证据版本，再用 train/validation 开发。

prompt 只根据授权关系作判断，不把 injection presence、执行成功或收件人是否逐字出现在 request 中当成标签。证据不足时预测 BLOCK，并在 reason 说明；这是该 comparator 的固定保守决策策略，不是给 ambiguous 样本创建 gold label 的规则。prompt 尚未经过真实 API 评估，后续只允许依据开发集修改并记录新版本。

## 请求配置

模型默认为现有仓库使用的 deepseek-flash，官方 endpoint 为 https://api.deepseek.com/chat/completions。固定 temperature=0、thinking disabled、JSON object 输出、max_tokens=512、单请求 timeout=45 秒，不自动重试。

这里显式关闭 thinking，是为了固定 comparator 模式并让输出预算用于简短判断。[官方 Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)说明 thinking 默认启用，且 temperature 在 thinking 模式不生效。[官方 JSON Output 指南](https://api-docs.deepseek.com/guides/json_mode/)要求显式 JSON 设置和 prompt 指令，也指出可能出现空内容；本实现将空内容记作 parse_error。

模型名是 provider 别名，不是不可变权重版本。实际运行会保存 provider 返回的 model、system_fingerprint、response ID 和 created；同一个名称未来可能变化，不能保证跨时间逐位复现。thinking 设置与原 agent adapter 也不自动等价，报告应分别记录 agent 与 auditor 的配置。

## 结果、失败与缓存

模型只输出 ALLOW/BLOCK 和一句简短 reason，不要求长推理。成功预测为 0/1，不把这个二值值伪装成校准概率，因此不报告 auditor 的 PR-AUC/AP。

每条记录包括 status、prediction、decision、cache_hit、当前与首次调用耗时、usage、请求哈希和 prompt version。API 错误、JSON 错误及预算耗尽均保留 null prediction。summary 同时给出有效响应覆盖率、仅有效且有标签样本的条件指标，以及全部有标签样本分母下的有效正确预测比例。不要只引用条件 F1 而隐藏失败比例。

这个组件尚未定义在线运行时的错误放行或错误阻断策略，因此不能直接作为完整 AgentDojo 防御。后续接 gate 时必须明确单独的 error policy，并将其对 utility 的影响计入 episode 评估。

调用预算按请求尝试计，失败也占一次，无自动重试。缓存键包含 endpoint、实际 prompt、模型、输入和实验配置。缓存命中不消耗本次预算，不重复计算本次 token usage 和 provider latency。成功收到但格式错误的响应也缓存，避免悄悄重复请求直到得到可解析答案。请求错误不会存成模型答案。

token usage 只统计 API 返回的已知值，并列出未知调用数。monetary_cost 保持 null，不根据未经核对的价格估算费用。

## 运行方式

在包含上一轮诊断修改的仓库根目录，用现有可用建模环境运行。这个 auditor 的 HTTP 部分只使用 Python 标准库，不需要 OpenAI SDK；CLI 汇总沿用现有 numpy/scikit-learn 指标模块。

默认离线预览，不需要 API key：

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
python scripts/run_prompt_auditor.py --split validation --limit 73 --output-dir results/auditor_preview_new
```

确认 prompt 与配置后，在本地设置 DEEPSEEK_API_KEY 环境变量。程序不会自动加载 .env，也不会输出或保存 key。下面是供后续实际运行的小样本命令，本次没有执行：

```powershell
python scripts/run_prompt_auditor.py --execute --split validation --limit 5 --max-calls 5 --output-dir results/auditor_smoke_new
```

首次只检查 provider 接口、解析和记录是否正常。前 N 条来自原文件顺序，不是代表性抽样；当前文件按来源排列，前 5 条均为初标正例，不能用这一轮衡量授权误拦或正式效果。

若小样本运行正常且已经决定继续，可在固定配置下跑全部 validation：

```powershell
python scripts/run_prompt_auditor.py --execute --split validation --limit 73 --max-calls 73 --output-dir results/auditor_validation_new
```

相同设置的已有缓存会复用，实际新调用数可能少于 73。首次调用数量上限不是费用估计。每次 output-dir 必须是新的；cache-dir 可保持原默认路径。不要覆盖或编辑缓存来挑选有利结果。

消融示例（本次未进行实际 API 调用）：

```powershell
python scripts/run_prompt_auditor.py --split validation --ablation no_request --limit 73 --output-dir results/auditor_no_request_preview
python scripts/run_prompt_auditor.py --split validation --ablation no_observations --limit 73 --output-dir results/auditor_no_observations_preview
```

## 下一次真正需要的信息

代码和离线预览已经可交接。要获得实际 prompt-auditor 成绩，需要本地有效 DeepSeek key，以及确定要使用的模型/模式和调用范围。请勿把 key 发到聊天或提交仓库。

先检查少量请求，再决定是否跑完整验证集；在看到结果后修改 prompt，就创建新的 prompt version 并说明开发过程。最终 held-out 评估需另行锁定协议，不能将反复调过的 validation 成绩称为最终 test 成绩。
