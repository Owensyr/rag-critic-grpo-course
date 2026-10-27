GRPO CLM：直接全量 2Wiki 推理及 Kaggle 完整回答导出
=================================================

这一步是已训练模型的推理，不重新做 GRPO。
使用已审查的 clm_rl_grpo_pilot_v3/artifacts/checkpoints/step016，加载到原 SFT CLM 底座。
复用已有 Qwen 的 12,576 条初始回答，每题一次 CLM 批判、一次 Qwen 修订。
answer 保存 Qwen 修订后的完整原文，不做短答案提取，不加训练用 final_answer 标签。

按照用户选择，直接全量：不要求先跑五题，不做概率差异探测。
仍检查权重 SHA256、SFT 底座来源、LoRA 注册与显式请求、结果行数/ID/顺序、CSV 原文读回一致性。
注册证明适配器可用，显式请求记录加载路径；这些检查不等于准确率提升证明。
前一次五题探测的实际结果留在 reference/prior_probe_audit.json，status 仍为 inconclusive，未改成 passed。

服务器前提
原来的四张 RTX 3090 空闲。模型、数据、环境沿用原服务器。
使用 ~/miniconda3/envs/rag-author/bin/python（vLLM 0.10.2），不要改用 base Python。
无需重装环境、下载权重或合并一份新 8B 模型。vLLM 按请求加载 RL LoRA。
官方用法：https://docs.vllm.ai/en/v0.10.2/features/lora.html

1. 上传 clm_rl_step7_2wiki_v3.tar.gz 到 ~/RAG/。
旧 step7 和 step7_v2 目录保留，新目录不复制旧 artifacts。

cd ~/RAG
tar -xzf clm_rl_step7_2wiki_v3.tar.gz
cd clm_rl_step7_2wiki_v3
mkdir -p logs

2. 直接后台运行全量（程序自动做输入和环境检查）

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 ~/miniconda3/envs/rag-author/bin/python -u run.py full > logs/full.log 2>&1 < /dev/null &

3. 查看状态

~/miniconda3/envs/rag-author/bin/python run.py status --mode full
tail -f logs/full.log

启动时读取校验约 30GB 模型文件，可能耗时数分钟。
stage=phase2：SFT CLM + RL LoRA 生成批判。
stage=phase3：冻结 Qwen 根据批判修订原回答。Qwen 本身不挂载 RL LoRA。
各阶段详细日志：artifacts/rl_full/phase2.log、artifacts/rl_full/phase3.log。

成功条件
status=complete；stage=evaluation_complete；completed_phases=[2,3]；rows=12576。
日志出现 RL_2WIKI_COMPLETE rows=12576 csv=...
不以报告包是否存在判断成功，因为失败也会打包报告。

4. 跑完下载
提交 CSV：
~/RAG/clm_rl_step7_2wiki_v3/artifacts/rl_full/submission_qwen_rl_clm_full_rationale.csv
审查报告：
~/RAG/clm_rl_step7_2wiki_v3/clm_rl_step7_v3_rl_full_reports.tar.gz

两者下载到本地 RAG/server_results/复现/CLM/。Kaggle 只上传 CSV。
建议备注 Qwen + SFT CLM + GRPO step016, full rationale。
与原 Qwen + SFT CLM 的完整回答提交比较；不与早期短答案提取提交混为一组。

报告中保留
批判全文、修订全文、CSV、phase2_runtime.json（LoRA 路径、ID、注册列表、请求数）、phase3_runtime.json、run_manifest.json 和日志。
vLLM V1 返回 lora_request 字段为空时如实记录 missing，不据此推断加载失败。
同一运行只有返回相矛盾的 ID/路径、注册失败、权重来源错误等真实错误会停止。

恢复和可选操作
重新运行同样的 run.py full 命令可恢复：已完成阶段校验后跳过，未完成阶段从头执行。
不得并行启动多个 GPU 任务；本包有运行锁，但不会停止其他项目的 GPU 进程。
手动重新打包：~/miniconda3/envs/rag-author/bin/python run.py report --mode full
五题预览仍可自选运行 run.py smoke；它不是 full 的前置条件。
如需要完全相同的新引擎配置下的 SFT 对照，可另外运行 run.py full --variant sft，输出单独保存；本次不自动运行。

记录的比较边界
提示词、前五篇文档顺序、原始 Qwen 回答、一次批判修订、完整回答 CSV 协议不变。
critic 温度 0.6/top_p 0.9；Qwen 0.7/0.8；seed 42；max_tokens 5120；max_model_len 32768。
四卡 TP=4，max_num_seqs=32，eager，关闭 prefix cache，底座和运行时 LoRA 使用 BF16。
新 LoRA 内核和批处理设置可能与旧 SFT 运行不同；不保证逐字相同，也不预先声称 Kaggle 一定提升。
课程测试集无标签，不参与训练或自动调整检查点。
