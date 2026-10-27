第五步：第四步原始10K CFT语料，LLaMA-3.1-8B-Instruct LoRA训练

原始文件固定SHA256：c500875fc40020ea3fe302e8797deb4512a46c92e54ab5150d52e16f9928de72
默认路径~/RAG/clm_training_step4_data/artifacts/full/cft_train_10k.json。
不使用clm_repair137产物，不因答案不匹配、重复、生成截断而删除或重生成样本。
保留已记录的256道生成截断题，其中137道涉及训练目标；这是用户明确选择的主实验。
训练只更新LLaMA LoRA；Qwen7B基线不变。最终只在2Wiki测试，训练本步骤不访问测试集。

来源
作者固定提交11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a：
https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/train/train_llama_rationale.yaml
论文附录训练参数：rank16/alpha64、AdamW、lr1e-5、2epochs、warmup0.1、bf16、cutoff6144。
https://proceedings.neurips.cc/paper_files/paper/2025/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf#page=24
本包使用LLaMA-Factory 0.9.3，API核对版本ca75f1edf3cb50343ed1c98605141c3e22075b5f。
https://github.com/hiyouga/LlamaFactory/tree/ca75f1edf3cb50343ed1c98605141c3e22075b5f
作者没有固定LLaMA-Factory版本；本版本是本次复现的明确选择。

实际配置
底座：LLaMA-3.1-8B-Instruct，bf16非量化LoRA，lora_target=all，rank16，alpha64，dropout0。
lr1e-5，AdamW(torch)，cosine，warmup_ratio0.1，2epochs，cutoff6144，seed42。
训练代码沿用作者LLaMA-Factory SFT/CFT路线；没有自行实现优化器或改变监督目标。
作者YAML val_size=.1：10K语料随机划分9000训练、1000内部验证，split seed42。
按LLaMA-Factory默认Alpaca映射：instruction和input用换行拼接为USER，output为ASSISTANT。
不把instruction擅自另设为SYSTEM；只对ASSISTANT目标计算loss。
预处理调用固定版本的AlpacaDatasetConverter与SupervisedDatasetProcessor。
逐条核对完整编码和label mask，要求无6144截断；实际编码检查通过才训练。
已编码的train/validation缓存固定保存，因此训练config中的val_size=0避免再次划分。
这1000条验证不是Kaggle测试集，eval_loss不等于最终2Wiki准确率。

硬件与配置差异（需写入实验报告）
原论文2xA100 80GB，本机4x3090 24GB。
每卡batch1、gradient_accumulation_steps8、四卡DDP，有效batch32。
论文未给明确梯度累积数，无法断言有效batch与论文完全相同。
作者YAML每卡batch4、accum4、cutoff20480、DeepSpeed ZeRO2 offload；
本包cutoff遵论文6144，使用DDP、SDPA及gradient checkpointing，避免额外DeepSpeed编译依赖。
保留两轮训练、核心LoRA参数与目标格式。非QLoRA，不修改模型参数量。
wandb关闭，日志与模型保存在本地；每100步保存checkpoint，最多保留2个。
ASQA部分支持分层等已确认的数据重建差异继续保留，不声称原作者10K样本逐条复现。

环境
从rag-author启动setup_env.sh，创建~/RAG/clm_training_env独立venv。
通过--system-site-packages复用原torch2.8.0+cu128，避免重下CUDA/PyTorch。
在新venv内固定训练依赖，Transformers4.51.3、LLaMA-Factory0.9.3等。
不会在rag-author中安装/降级包；后续vLLM推理仍回rag-author运行。
新环境继承但不用于训练的vLLM可能有版本约束冲突提示；脚本核对实际训练依赖。
pip使用默认配置或用户设置的PIP_INDEX_URL，证书验证保持开启。

启动（确认四张卡空闲，建议可用磁盘至少10GB）
上传clm_training_step5.tar.gz到~/RAG后：
cd ~/RAG
tar -xzf clm_training_step5.tar.gz
conda activate rag-author
cd ~/RAG/clm_training_step5
mkdir -p logs
set -o pipefail
bash setup_env.sh 2>&1 | tee logs/setup.log

看到TRAINING_ENVIRONMENT_OK后：
nohup ~/RAG/clm_training_env/bin/python -u run.py all > logs/pipeline.log 2>&1 < /dev/null &

run.py all依次执行
1. 校验包、模型config和原始CFT哈希；以真实LLaMA-Factory格式编码全部10K并固定9K/1K划分。
2. 取训练划分中最长64条，进行3个优化步短测（相同batch/梯度累积），并对8条做验证。
3. 检查finite loss/梯度日志、四卡峰值、LoRA_B确实更新及adapter可读取。
4. 短测通过后，重新从原始LLaMA底座开始两轮正式训练，不继承短测权重。
5. 验证最终adapter非NaN且已更新、epoch达到2，再记录完成。
任一步失败即停止，不自动降精度、减数据或更换训练方法。
本机无法运行四卡3090实测，实际显存是否足够由短测确定。

查看进度
~/RAG/clm_training_env/bin/python run.py status
tail -n 30 logs/pipeline.log
tail -n 30 logs/smoke.log
tail -n 30 logs/full.log
full.log只有正式训练启动后才存在。当前任务的loss/进度在对应phase日志。
成功标记CLM_LORA_TRAINING_COMPLETE epochs=2。
如果失败，不要直接覆盖输出目录；先打包日志。存在完整checkpoint时可用：
nohup ~/RAG/clm_training_env/bin/python -u run.py all --resume > logs/pipeline_resume.log 2>&1 < /dev/null &
有完整阶段验证报告时会跳过已完成阶段；不同时启动多个训练任务。

输出
artifacts/full/adapter_model.safetensors：正式训练的LoRA权重，不是完整8B模型。
artifacts/full/adapter_config.json及tokenizer文件：加载adapter所需配置。
artifacts/full/checkpoint-*：恢复训练所需checkpoint和优化器状态。
artifacts/full/trainer_state.json、trainer_log.jsonl、train_results.json、eval_results.json、loss图。
artifacts/*/gpu_rank*.json：每张卡训练参数和显存峰值。
reports/prepared.json、split_indices.json、tokenization_preview.json：数据、划分、真实编码证据。
reports/environment.json、training_environment_freeze.txt：环境。
reports/smoke_validation.json、full_validation.json、pipeline.json：验证及整体状态。
configs/*.json：实际执行的训练配置。

回传
不管成功还是失败：
~/RAG/clm_training_env/bin/python collect_reports.py
下载~/RAG/clm_training_step5/clm_step5_training_reports.tar.gz。
若环境安装失败可直接用rag-author的python collect_reports.py，工具仅依赖标准库。
报告包不含权重、优化器和tokenized缓存。模型权重留服务器，建议自行备份。
本包不自动合并或部署，训练报告检查后再合并adapter并进行2Wiki推理。
