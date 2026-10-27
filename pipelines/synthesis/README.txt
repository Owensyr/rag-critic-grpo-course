第四步：10K 分层训练输入 + 批量强弱回答及对比批判生成

本包已经包含本地构造并核验的 train_10k.jsonl，无须向服务器上传五个完整原始训练集。
它是按论文规则重建的数据，不是作者发布的原始10K样本或训练完成的CLM。
当前不启动LoRA训练。沿用rag-author环境和已下载模型，不需要安装新依赖。

依据
论文附录A.3（第25页）：每个数据集200条随机无关文档、400条top5无答案，
以及支持文档数1/2/3/4/5各400/400/200/200/200条；五数据集总共10K。
https://proceedings.neurips.cc/paper_files/paper/2025/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf
固定作者提交：11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a
https://github.com/upup-wei/RAG-ReasonAlignment/tree/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a
author/保留原始提示词、采样YAML、任务后缀、CFT格式文件和metrics.py，供对照。

必须记录的重建差异
1. ASQA：用户于2026-10-23选择“任一子问题答案出现”作为文档支持指标，
   对应论文包含部分支持证据的描述；这是我们的操作化解释，不是原代码完全等价实现。
   原代码要求单篇覆盖所有子问题，其3/4/5档有效候选只有98/60/47，无法各取200。
   每条同时保留support_flags和official_support_flags。官方全覆盖统计不会隐藏。
2. 其他数据集使用作者normalize_answer + 子串出现判定的等价实现，匹配标题与正文。
   yes/no等答案可能偶然匹配，计数不等同于语义蕴含或人工验证。
3. 固定random seed=42抽样，不是sample.py直接取前2000条。跨已选数据集去重，
   排除五个作者test和课程test的规范化问题重叠；不进行语义改写级查重。
4. H1从同数据集其他合格训练题的文档抽取：排除目标原文档、目标答案子串匹配，
   并要求双方问题去除常见词后的词集合不交叠。这是可复现的无关文档近似，
   仍需抽查语义相关性； donor原始训练索引逐条记录。
5. 保留原检索结果中的重复文档，不人为复制文档凑配额；build_report记录重复数。
6. 原训练文件自带rationale没有作为监督使用。所有强弱回答和批判重新生成。
7. 作者提示词/任务后缀与CFT模板保持原文。路径、逐批保存、离线运行等由run.py适配。
   检索score转float；按作者规则反转前5篇顺序；一次chat模板，不二次加入BOS。
8. temperature=0.6、top_p=0.9、max_tokens=1024来自作者YAML。
   本批量运行逐条seed=42+固定行号，保证断点重试时有固定的采样种子；GPU计算不保证位级复现。
   超过8192输入token时停止报错，不静默裁掉证据。输出触顶会保留并计数，等待质量审查。

上传和执行
上传clm_training_step4_data.tar.gz到~/RAG，确认四张卡空闲：
cd ~/RAG
tar -xzf clm_training_step4_data.tar.gz
conda activate rag-author
cd ~/RAG/clm_training_step4_data
python run.py check
mkdir -p logs
nohup python -u run.py all > logs/full.log 2>&1 < /dev/null &

默认模型
~/RAG/models_verified/Qwen/Qwen2.5-0.5B-Instruct
~/RAG/models_verified/meta-llama/Llama-3.1-8B-Instruct
可用--weak和--strong指定目录。弱模型只占GPU0，强模型和批判均使用GPU0,1,2,3。
三个子进程顺序执行并释放显存；不要同时运行其他GPU任务。

进度和恢复
python run.py status
tail -n 30 logs/full.log
grep 'PROGRESS\|START_ROLE\|CLM_CFT_SYNTHESIS_COMPLETE' logs/full.log | tail -n 15
每批32条刷盘；正常中断后可用相同命令恢复（日志建议full_resume.log，保留原日志）。
flock防止同一输出目录同时启动两个主任务。
参数/代码/模型config/环境版本变化将拒绝复用检查点。未完成角色依赖已完成角色文件哈希。
突然断电造成JSONL末行损坏时会报错；保留文件并返回日志，不要直接删掉全部进度。

成功标记
CLM_CFT_SYNTHESIS_COMPLETE rows=10000 training_started=false
run_manifest.json status=complete；三个角色各10000条。

输出artifacts/full
weak.jsonl / strong.jsonl / critique.jsonl：三个阶段完整输出和终止原因、token数、提示词哈希。
weak.manifest.json / strong.manifest.json / critique.manifest.json：阶段参数、指纹与进度。
cft_train_10k.json：候选CFT监督集（instruction/input/output），还未质量审批。
review_preview70.json：各数据集每层2条，共70条，含文档、标签及全部生成内容。
quality_summary.json：长度截断、异常结束与答案出现诊断计数；不代表语义准确率。
run_manifest.json：整体状态和输出哈希。
不按“强模型命中答案”静默筛掉记录，不人工改写答案，也不认为强回答天然正确。

完成后打包回传（不包含模型权重和完整CFT大文件）
tar -czf clm_step4_reports.tar.gz logs data/build_report.json data/selection_preview70.json source_manifest.json artifacts/full/run_manifest.json artifacts/full/weak.manifest.json artifacts/full/strong.manifest.json artifacts/full/critique.manifest.json artifacts/full/quality_summary.json artifacts/full/review_preview70.json
下载~/RAG/clm_training_step4_data/clm_step4_reports.tar.gz给助手检查。
完整jsonl和cft_train_10k.json留在服务器用于后续训练，建议自行备份。
若失败，使用：tar -czf clm_step4_failed_reports.tar.gz logs artifacts/full data/build_report.json

本地验证范围
10K配额、问题去重、标签匹配、源数据哈希、数据传输哈希、生成结果导出与断点校验逻辑。
本机未运行本包的GPU推理；前两步相同模型/环境/原始提示词已在服务器通过5题预览。
真正开始LoRA前仍需检查训练token长度、四卡显存及全量语料质量。
