# 来源与授权范围

1. 本项目的 RAG 初始回答、批判、修订提示词，以及部分数据合成与训练模板，来源于 [upup-wei/RAG-ReasonAlignment](https://github.com/upup-wei/RAG-ReasonAlignment)，固定提交 `11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a`。课程适配保留 ID、允许无标签测试数据、转换文档分数类型并补充可移植的日志导入。
2. SFT 使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)；LoRA 通过 PEFT；推理通过 vLLM；对应软件沿用各自许可证。本仓库不重新分发这些软件的安装包。
3. GRPO 数学参考 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。这里的“批判策略 → 冻结 Qwen 修订 → 最终答案奖励”训练程序是本课程实验的实现，不是 RAG-ReasonAlignment 作者提供的 GRPO 实现。
4. [DeepRAG](https://arxiv.org/abs/2502.01142) 作为奖励设计思路参考；最终实验采用 outcome-only 奖励，没有实现检索次数惩罚，不能称为对其奖励公式的原样复现。
5. 模型权重分别遵循 [Qwen 模型页](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 与 [LLaMA 模型页](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 的许可和获取要求。课程数据与作者数据仍来自各自的数据来源。
6. 本次读取的上游仓库根目录未展示独立 LICENSE 文件。因此没有把整套含上游来源的代码擅自标注为本团队 MIT 授权；保留来源说明不等同于取得新的授权。
