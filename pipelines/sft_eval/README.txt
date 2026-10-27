第六步：已训练 CLM 合并与课程 2Wiki 评测

仅接受已审核的第五步最终 adapter，SHA256：
f0c8979757af7dffd41635eb61387902a7d367f73730c9bc31855cc2b8388343
底座为训练所用 LLaMA-3.1-8B-Instruct，绝不使用原仓库示例中的其他底座。
原始第四步 10K，保留用户选定的未过滤样本；不使用 repair137 结果。
报告审核结果见 training_review.json。低验证损失不代表 2Wiki 准确率提升。

合并沿用 LLaMA-Factory 0.9.3 export_model：CPU 加载，非量化 BF16 导出。
官方实现：
https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/train/tuner.py
https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/examples/merge_lora/llama3_lora_sft.yaml
需要额外约 16GB 权重磁盘空间，脚本要求至少 20GiB 空闲；CPU 合并也需要内存，
建议运行前 free -h 检查 available，预留至少约 32GiB，实际峰值随加载过程变化。
合并阶段使用 clm_training_env，推理阶段使用原 rag-author，脚本自动切换解释器。
无需安装新依赖。首次真实合并和 GPU 短测仍需在服务器验证。

上传 clm_step6_2wiki.tar.gz 到 ~/RAG，解压为新目录：
cd ~/RAG
tar -xzf clm_step6_2wiki.tar.gz
cd ~/RAG/clm_step6_2wiki
mkdir -p logs
free -h
df -h .
nvidia-smi
nohup bash merge_and_smoke.sh > logs/pipeline.log 2>&1 < /dev/null &

确认四张卡空闲后启动一次。CPU 合并期间显卡空闲正常。
如 conda 位置不同，可在启动时设置 INFER_PYTHON=/实际/rag-author/bin/python。
查看进度：
tail -n 30 logs/pipeline.log
tail -n 30 logs/merge.log
tail -n 30 logs/smoke.log
smoke.log 在短测开始后才出现。
整个合并加短测完成标记：CLM_STEP6_SMOKE_COMPLETE rows=5

合并后检查每个张量有限且 BF16、索引与底座键集合一致、三个层的 q_proj 确实改变、
tokenizer 可加载和聊天模板可编码。记录全部文件哈希，推理前重新核对。
模型目录必须包含 Llama：既有作者推理代码依据路径字符串选择采样参数。
模型保存为 models/Llama-3.1-8B-Instruct-CLM-original10k/。
若合并已完成，校验哈希后复用；若目录未完整写入则拒绝覆盖，请回传报告。

五题短测沿用全量 Qwen 基线的前五题，只运行 phase 2 批判和 phase 3 修订。
eval/ 和 configs/eval.yaml 与先前 Frozen CLM 包逐字节一致。
Qwen temperature 0.7/top_p 0.8；CLM temperature 0.6/top_p 0.9；
seed42、top5 文档、一次修订、max_tokens5120、max_model_len32768、四卡 TP。
输出完整修订回答，不做短答案提取。课程测试集12576题无标签，不能本地算准确率。

短测产物：
artifacts/smoke/run_manifest.json
artifacts/smoke/2WikiMultiHopQA/result_critic_rationale_cot.json
artifacts/smoke/2WikiMultiHopQA/result_refine_cot.json
artifacts/smoke/preview_qwen_trained_clm_full_rationale.csv（仅预览，不能提交）

短测结束自动打包 clm_step6_reports.tar.gz，请下载回传检查。
失败时也可手动：
python collect_reports.py
报告不含模型权重；模型留服务器。

短测确认后启动全量：
conda activate rag-author
cd ~/RAG/clm_step6_2wiki
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run.py full > logs/full.log 2>&1 < /dev/null &
tail -f logs/full.log
完成标记：TRAINED_CLM_COMPLETE rows=12576
生成 artifacts/full/submission_qwen_trained_clm_full_rationale.csv，用于手动提交 Kaggle。
全量结束再次运行 python collect_reports.py，回传报告包及全量 CSV。
全量完整批判/修订文本保存在 artifacts/full/2WikiMultiHopQA/，按需下载。

已有输出目录拒绝覆盖。没有自动断点续生成功能；需要重跑时可用
python run.py smoke --output artifacts/smoke_retry1（报告包自动收集默认 smoke，
使用自定义目录时请同时下载该目录）。已有模型不需要重新合并。
模型训练数据和硬件差异延续先前记录，不声称完全复现作者分数。
