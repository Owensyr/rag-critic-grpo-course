V3: Deployed hybrid scorer; original40 and paraphrase16 each require >=95%, all critical cases pass.
First run run.py reward-check. No GPU calibration has been run locally.
Use a NEW directory; do not copy old artifacts/spec.json or reward_cache.json.
Training data, SFT model, RL optimizer and inference settings are unchanged.

CLM8B GRPO pilot: 128 train + 128 held-out development questions
=============================================================

目标
从已训练并合并的 CLM-original10k 重新添加 RL LoRA，冻结 Qwen 与 SFT 基础权重。
固定 128 道验证题，比较 SFT 起点与第 16 轮后的模型。这里是小规模效果实验，
不是原论文训练代码，也没有自动启动 12,576 题的 Kaggle 全量评测。

上传和启动（四张 3090 空闲后）
把 clm_rl_grpo_pilot_v3.tar.gz 上传到 ~/RAG/，执行：

    cd ~/RAG
    tar -xzf clm_rl_grpo_pilot_v3.tar.gz
    cd clm_rl_grpo_pilot_v3
    mkdir -p logs
    set -o pipefail
    CUDA_VISIBLE_DEVICES=0,1,2,3 ~/RAG/clm_training_env/bin/python run.py check 2>&1 | tee logs/preflight.log

看到 GRPO_PILOT_ENVIRONMENT_OK 后执行：

    nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 ~/RAG/clm_training_env/bin/python -u run.py all > logs/pipeline.log 2>&1 < /dev/null &
    echo $!

使用已有 clm_training_env 与 rag-author 两个环境，不需要另装依赖或下载模型。
程序通过已配置的 bin/python 启动各阶段，保留虚拟环境入口，不解析解释器符号链接。
最低预留 12 GiB 磁盘；每轮保存 RL adapter 和 Adam 状态，基础 8B 权重不复制。
四卡 CLM 采样、四卡 Qwen 推理、四卡梯度更新依次运行，不同时占显存。

自动执行顺序
1. 校验代码/数据/模型来源、依赖、GRPO 数学与小型 LoRA 测试。
2. 检查实际部署的“严格规则 + 冻结 Qwen”评分流程，原 40 条和新增 16 条分别要求至少 95%，所有关键样例通过。
   纯 Qwen 对全部 56 条的评分另存为 raw_model_diagnostic，不宣称纯 Qwen 已通过校验。
   这些只是固定回归样例，不是模型在真实 2Wiki 上的评分准确率估计。
3. 对 256 道题缓存 Qwen 原始回答，使用作者原始初始回答提示词。
4. SFT CLM 在 128 道验证题上生成贪心批判，Qwen 修订并评分。
5. 16 轮在线 GRPO，每轮 8 道新训练题，每题 4 个温度 1 的批判候选。
6. 最后一轮模型在同一 128 题上评测；生成逐题配对比较与报告。

只检查评分器或只跑训练前基线时，也支持：
    CUDA_VISIBLE_DEVICES=0,1,2,3 ~/RAG/clm_training_env/bin/python -u run.py reward-check
    CUDA_VISIBLE_DEVICES=0,1,2,3 ~/RAG/clm_training_env/bin/python -u run.py baseline
之后 run.py all 会跳过已完成且哈希一致的阶段。

进度
    ~/RAG/clm_training_env/bin/python run.py status
    tail -f logs/pipeline.log

pipeline.log 显示阶段起止；status 同时显示当前阶段日志最后 6 行。
详细日志示例：logs/round1_critic.log、logs/round1_qwen.log、logs/round1_score.log、logs/round1_update.log。
ROLLOUT rank=... question=... 表示当前批次/验证集内进度，不是全局训练题编号。
最终成功标记：
    GRPO_PILOT_COMPLETE optimizer_steps=... dev_questions=128
status 中应为 status=complete，stage=pilot_complete。
实际 optimizer_steps 可能少于 16，因为全同奖励的一轮不会执行更新。

恢复与停止
重跑完全相同的 run.py all 命令即可恢复。完成阶段校验哈希后跳过；未完成的
CLM 采样按已写出的题目恢复；未完成 Qwen 生成阶段重新计算；评分调用有缓存。
更新阶段若中断，从上一轮已完成 checkpoint 重算，不继续半次梯度更新。
不要改动已有运行目录的代码、配置、数据或模型再续跑，程序会拒绝混合版本。
若需要停止，用 run.py status 查看本任务 pid，再对该 PID 执行 kill -TERM。
不要用 killall python；主进程会终止自己启动的子进程并尝试打包报告。
因奖励校验、截断率、KL 等检查停止时，先下载报告检查，不要自行关闭检查后继续。

奖励协议 outcome_only_hybrid_v5
- Qwen 修订后选定的答案正确：1；错误：0。没有格式加分、检索次数或长度惩罚。
- 精确匹配、无歧义日期及有限的国家别名用规则处理。
- 整句可严格识别的两实体国家关系用真值表转换为问句对应的 yes/no，再与标准答案比较。
  不识别含转折、候选、时态不一致、实体不一致的句子；这些仍走语义评分。不得用子串命中奖励。
- 带/不带 Jr./Sr./II 等后缀且没有明确别名支持的同名实体记为 unresolved，不自动视作同一个人。
- 其他表达使用冻结的 Qwen 判断“问题、标准答案、所选答案”是否等价。
  这是自定义的模型评分器，没有使用 General-Verifier 权重，也不把它当成完美裁判。
- 最终答案标签缺失/不规范时，先做不含标准答案的忠实提取，要求提供原文引用。
  不根据标准答案从原回答里寻找碰巧出现的正确词。
- Qwen 评分不确定、解析失败或生成截断，均单独记为 unresolved；不伪装成错误。
- 一道训练题的四个候选只要有无法评分或截断，该题整组不参与更新。
- 可用组少于当轮一半，或批判截断率超过 25%，会停止并保存报告。
- 全对/全错组的正确性优势为零。整轮都没有差异时，原样保留 adapter 和 Adam 状态。
  连续三轮无差异则停止检查。少数零优势组可参与有效批次的 KL 约束。
- 标准答案仅在独立评分进程使用，不进入 Qwen 回答或 CLM 批判的输入。
- 评分原始提示和输出保存在 artifacts/reward_cache.json，可审计错误判定。

训练设置
新 LoRA rank=8，alpha=16，覆盖 attention 和 MLP 的七种线性投影。
学习率 1e-6，KL 系数 0.04，clip epsilon=0.2，最大梯度范数 1。
CLM 采样温度 1、top_p=1、top_k=0；Qwen 回答和评分均为温度 0。
CLM 最大新生成 1536 token，总长度上限 4096；不静默截断提示词。
每轮采用一次优化更新；更新前策略等于采样策略，因此该步裁剪通常不触发。
批判完成部分（含生成的 EOS）计算损失，问题/文档/Qwen 输出不计算策略损失。
四卡梯度显式求平均；每轮核对旧 logprob、冻结参考及四副本一致性。
SFT 基础模型为固定 KL 参考，通过禁用新增 RL adapter 获得。

数据与评测
已在本地从作者 2Wiki train 的 167,454 道题中准备数据，无需服务器再次扫描大文件。
排除 SFT 10K、此前两次 RL 开发样本共 64 题，以及作者五个测试集和课程测试集。
去重是规范化问题文本相等，不能保证不存在语义改写。抽样未根据模型回答筛选。
本次 Qwen 修订提示附带 final_answer 字段；训练前后使用完全相同的设置。
该提示词与原始 Kaggle 全文评测有差异；不能把这里的验证正确率直接当 Kaggle 分数。
这次固定比较最后一轮，没有用验证集选“最佳”checkpoint，也没有使用测试标签训练。
验证报告保留完整分母、未判定题数、准确率上下界、错变对/对变错和配对 bootstrap 区间。
单次种子、128 道验证题及模型评分器均有限制；最终结论需要检查逐题结果，并在
协议固定后用原来的完整回答提交流程验证。

服务器保留的文件
artifacts/checkpoints/step000/             新建零增量 RL adapter（SFT 起点）
artifacts/checkpoints/step001/...step016/ 每轮后的 RL adapter 和 Adam 状态
artifacts/initial.jsonl                   冻结 Qwen 原始完整回答
artifacts/dev_before/                     SFT 验证批判、Qwen 完整修订、评分
artifacts/round1/...round16/               训练候选、Qwen 修订、奖励、梯度和资源记录
artifacts/dev_after/                      GRPO 验证输出
reports/reward_calibration.json           56 条评分器检查结果
reports/summary.json                      训练前后指标与训练摘要
reports/paired_dev.jsonl                  逐题前后对照
clm_rl_grpo_pilot_v3_reports.tar.gz           自动生成的报告包

需要下载给助手检查
只下载 clm_rl_grpo_pilot_v3_reports.tar.gz 到本机：
D:\研究生\人工智能原理与应用\RAG\server_results\复现\CLM\
报告包含代码、配置、数据、日志、完整回答与评分，不包含模型权重或 optimizer.pt。
正常结束/异常退出都会尝试打包。也可手动执行：
    ~/RAG/clm_training_env/bin/python run.py report

实现来源与界限
GRPO 组内标准化、token 裁剪及参考 KL：
https://arxiv.org/html/2402.03300v3
模型判定答案等价性的研究参考：
https://arxiv.org/abs/2505.14652
本实现把 CLM 当作待优化策略，把冻结 Qwen 修订结果用于反馈；这是针对本项目的适配。
本地已验证协议/恢复逻辑及 CPU 损失数学；真实 Qwen 评分器和四张 3090 尚需服务器检查。
