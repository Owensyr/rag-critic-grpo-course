沿用已提交 Kaggle 的完整 Qwen 回答，运行一次 Frozen CLM 批判与修订

步骤：原版 LLaMA-3.1-8B-Instruct 对原 Qwen 回答给出批判；Qwen2.5-7B-Instruct
结合问题、原有5篇文档、原回答和批判生成修订回答。最后将整段修订回答原样写入 CSV。
不重新运行 phase 1，不调用短答案提取器，不使用作者的有标签测试集。
本实验是 Frozen CLM 对照，不等于论文中训练后的 CLM 主方法。
Kaggle 成绩由人工上传获取；没有在本地推断或验证比赛内部评分实现。

输入使用服务器已有文件，并校验与旧提交来源一致的 SHA256：
~/RAG/author_alignrag_course/course/artifacts/full/qwen/2WikiMultiHopQA/result_rationale_cot.json
~/RAG/author_alignrag_course/course_data/2WikiMultiHopQA/test.json
~/RAG/author_alignrag_course/course_data/2WikiMultiHopQA/sample_submission.csv
模型默认路径：
~/RAG/models_verified/Qwen/Qwen2.5-7B-Instruct
~/RAG/models_verified/meta-llama/Llama-3.1-8B-Instruct
如路径不同，使用 --qwen、--llama、--baseline、--test、--sample 覆盖。
LLaMA 必须下载完整且为原版 Instruct 权重。此包不包含权重和数据。

将 frozen_clm_full_answer.tar.gz 上传到 ~/RAG/，确认同名目录尚未存在后解压：
cd ~/RAG
tar -xzf frozen_clm_full_answer.tar.gz
conda activate rag-author
cd ~/RAG/frozen_clm_full_answer
python run.py smoke --check-only

以上仅检查数据和源码，不代表模型或GPU可用。
确认旧任务已经结束、4张卡空闲且LLaMA下载完整后运行：
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run.py smoke

smoke 使用旧全量基线的前5条回答。先LLaMA后Qwen，分两个独立子进程顺序执行。
成功标记：FROZEN_CLM_COMPLETE rows=5。
完整文本：artifacts/smoke/2WikiMultiHopQA/result_critic_rationale_cot.json
及 artifacts/smoke/2WikiMultiHopQA/result_refine_cot.json。
5题预览 CSV 不能提交 Kaggle。

5题成功后运行全量（模型非默认路径时也须在此命令追加 --llama）：
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run.py full > logs/full.log 2>&1 < /dev/null &
echo $!
tail -f logs/full.log

START_PHASE_2：LLaMA生成批判。START_PHASE_3：Qwen生成修订。
每个阶段的 Processed prompts 单独从0计数到12576。
FROZEN_CLM_COMPLETE rows=12576 才代表两阶段及CSV校验全部成功。
下载并手动提交：artifacts/full/submission_qwen_llama_full_rationale.csv
运行记录及哈希：artifacts/full/run_manifest.json
生成文本：artifacts/full/2WikiMultiHopQA/ 下的两个 JSON 文件。
CSV保留原始换行和引号，必须用CSV解析器数记录，不能用wc -l计题数。
原始Qwen基线及其提交文件不会改动。

输出目录已存在时脚本拒绝覆盖。失败记录保留在run_manifest.json和日志。
没有自动断点续生成功能；需要重跑时使用 --output artifacts/full_retry1。
若只有phase3失败，可保留成功的批判结果，另行安排只重跑phase3。

沿用course推理适配代码，提示词、文档处理及评分函数与本地official副本一致。
参数来自作者Python代码：seed42，top5，max_tokens5120，max_model_len32768，bf16，
4卡张量并行，Qwen temperature0.7/top_p0.8，LLaMA temperature0.6/top_p0.9。
源码哈希记录于source_manifest.json，GPU实际运行效果须以服务器输出为准。
