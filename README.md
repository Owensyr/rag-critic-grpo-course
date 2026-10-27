# 基于批判模型与 GRPO 的 RAG 问答改进

人工智能原理与应用课程实验：固定 Qwen2.5-7B-Instruct 为回答模型，在课程版 2WikiMultiHopQA 上比较 Qwen 基线、Frozen CLM、SFT CLM 与 SFT CLM + GRPO。

**本 README 是从基础模型开始的完整训练路线，不需要下载我们的 `course_private_assets.tar.gz`、SFT LoRA 或 RL LoRA。** 读者自行下载基础模型和作者公开预处理数据，从课程来源获取测试集及提交模板，然后构造训练数据、训练模型并生成 Kaggle CSV。基础模型权重仍然必需；代码不会从随机初始化训练 7B/8B 模型。

## 历史实验结果

| 组别 | 回答模型 | 批判模型 | Kaggle 分数 | 相对上一组 |
| --- | --- | --- | ---: | ---: |
| Qwen 基线 | Qwen2.5-7B-Instruct | 无 | **0.45507** | — |
| Frozen CLM | 同上 | 原版 LLaMA-3.1-8B-Instruct，未做本项目微调 | **0.46382** | +0.875 个百分点 |
| SFT CLM | 同上 | 五数据集 10K 批判语料微调的 CLM8B | **0.51311** | +4.929 个百分点 |
| SFT CLM + GRPO | 同上 | CLM8B 上新增 GRPO LoRA，step016 | **0.51523** | +0.212 个百分点 |

分数由实验执行者于 2026-10-27 提供，见 [成绩记录](results/kaggle_scores.json)。四组都提交 12,576 条完整 Qwen 回答。没有取得 Kaggle 内部评分代码；新训练的权重、采样输出和成绩不保证与历史运行相同。特别是新的 RL 选题不依赖我们早期 reward32/smoke 的私有选题文件，因此不是历史 pilot 的逐题重放。

![四组历史成绩](results/kaggle_scores.png)

基线使用数据中提供的检索文档，并非闭卷回答。后三组各进行一次批判、一次修订，然后按固定预算结束；本实验没有在线重新检索或质量驱动的自适应停止。GRPO 的小幅提升是单次观察，不构成多种子显著性结论。

## 1. 环境与目录

历史 GPU 配置：**Linux、4 × RTX 3090 24 GB**。当前入口仍要求四张可见 GPU。建议主机至少 64 GB 内存；数据选择会载入单个作者 train 文件，CPU 合并 8B 模型也需要较多内存。基础模型、合并权重、16 轮 RL checkpoint 和中间回答需额外磁盘空间，建议预留约 100 GB 并按实际运行增加。

以下命令都在仓库根目录执行。将第一行 URL 换成实际仓库地址：

```bash
git clone <本仓库地址> rag-critic-grpo-course
cd rag-critic-grpo-course
bash tools/setup_envs.sh
cp configs/paths.example.json configs/paths.json

export INFER_PY="$HOME/.venvs/rag-inference/bin/python"
export TRAIN_PY="$HOME/.venvs/clm-training/bin/python"
```

`setup_envs.sh` 需要系统已有 Python 3.10，分别建立推理环境和训练环境；不要混装两套 Transformers。每次打开新终端重新设置上面的两个变量。若使用现有环境，编辑 `configs/paths.json` 中的解释器路径，同时设置这两个变量指向同一组解释器。

默认配置中的路径相对仓库根目录，支持绝对路径和 `~`：

```text
models/                           自行下载的三个基础模型
data/raw/author/                  作者五数据集，工具解压生成
data/raw/course/                  从课程来源获取的测试集与模板
data/prepared/course/             经过校验的课程输入
data/prepared/sft/                本次选出的 10K 训练问题
data/prepared/grpo/               本次选出的 128 train + 128 dev
runs/synthesis/                   弱回答、强回答、批判合成
runs/sft_train/                   SFT 训练、适配器、日志
runs/models/CLM8B-SFT/             合并后的 SFT 完整权重
runs/grpo/                        GRPO 训练、step016 适配器、开发集报告
runs/*_full/                      四组评测输出及 CSV
```

`data/`、`models/`、`runs/` 和本机 `configs/paths.json` 都被 Git 忽略。`pipelines/` 是历史源码快照；工具复制到 `runs/` 后绑定新数据和模型哈希，不直接改写历史源码。

## 2. 下载基础模型

需要 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)，以及用于合成弱回答的 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。LLaMA 使用获准访问该模型的 Hugging Face 账号。已有完整模型可直接修改配置路径，跳过对应下载。

```bash
"$INFER_PY" -c 'from huggingface_hub import login; login()'
"$INFER_PY" tools/download_models.py --config configs/paths.json --include-weak
```

默认下载各官方仓库的 `main`。固定版本时可分别执行，例如：

```bash
"$INFER_PY" tools/download_models.py --config configs/paths.json --only llama --revision <模型commit>
```

不要把访问 token 写入命令文件或提交到 GitHub。后续训练与推理采用本地离线加载。

## 3. 获取数据并构造训练选题

### 3.1 作者公开预处理数据

从 [作者 README 的 Datasets](https://github.com/upup-wei/RAG-ReasonAlignment#0-datasets) 获取含检索文档的五数据集。下面使用作者链接对应的 Google Drive 文件 ID；若链接失效或达到下载配额，可在浏览器从作者页面下载同一 ZIP 并上传服务器，不需要我们的私有包。

```bash
"$INFER_PY" -m pip install -r requirements/data.txt
mkdir -p data/downloads
"$INFER_PY" -m gdown 1MVkdc4g9_D4REtaBFKeJ9gMun4qzdQtO \
  -O data/downloads/author_dataset.zip --continue

python3 tools/unpack_author_data.py \
  --archive data/downloads/author_dataset.zip --output data/raw/author
```

解压工具只提取五个数据集各自的 `train.json`、`test.json`，校验完整文件集合，并记录 ZIP 和各数据文件的 SHA256。已有解压文件也可放到以下结构，跳过下载、解压：

```text
data/raw/author/{2WikiMultiHopQA,ASQA,NaturalQuestions,PopQA,TriviaQA}/{train,test}.json
```

必须使用作者预处理的 `question/answers/ctxs` 数据；ASQA 还需要 `qa_pairs`，检索文档需要 `title/text/score`。上游训练文件已有的 rationale 不直接作为我们的监督目标，后面会重新生成。

### 3.2 课程 Kaggle 测试集

**课程测试集与作者 test 不是同一个文件。** 请从你有权限访问的课程/Kaggle 数据页面下载 `test.json`、`sample_submission.csv`；当前没有可验证的公开课程下载 URL，仓库不会虚构自动下载地址。

把两个文件上传服务器后执行，替换前两条命令中的本机源路径：

```bash
mkdir -p data/raw/course
cp /path/to/course/test.json data/raw/course/test.json
cp /path/to/course/sample_submission.csv data/raw/course/sample_submission.csv

python3 tools/prepare_data.py course \
  --test data/raw/course/test.json \
  --sample data/raw/course/sample_submission.csv
```

工具检查 12,576 条、唯一 ID、模板顺序和无答案标签，保留每题前五篇检索文档的 `title/text/score`，移除其他元数据。输出到 `data/prepared/course/`。不要用作者 train/test 的标签为课程测试集补答案。没有课程数据访问权限时，可研究训练代码，但不能生成对应这场比赛的有效提交。

### 3.3 选择 10K SFT 问题和 GRPO train/dev

```bash
python3 tools/prepare_data.py sft
python3 tools/prepare_data.py grpo
```

SFT 选择沿用 `pipelines/synthesis/build.py`：五数据集各 2,000 条，按文档支持层级分层；ASQA 按“任一子问题答案出现”的部分支持口径。排除五个作者 test 和课程 test 的规范化同题，保留重复检索段落的历史处理方式。输出选题、分层审计、70 条预览。

GRPO 只从作者 **2Wiki TRAIN** 中均匀抽样 128 条训练题、128 条开发题；排除 SFT 选题及上述全部评测问题，并进行 train/dev 去重。问题和标签分开保存，prompt 文件不包含 `answers`、`has_answer` 或原始 rationale。若有自己之前用来开发奖励的题目，可额外传入 `--exclude-questions /path/to/old_questions.json`（可重复）。规范化去重不等同于语义去重。

每个准备命令要求输出目录不存在，防止覆盖旧实验；报错后保留诊断文件，重试时使用新的 `--output` 并在后续命令更新路径。

## 4. 合成 10K 批判训练语料

```bash
python3 tools/prepare_workdir.py synthesis --config configs/paths.json \
  --data-dir data/prepared/sft --accept-new-data --output runs/synthesis

nohup python3 -u tools/launch.py --workdir runs/synthesis \
  > runs/synthesis/logs/driver.log 2>&1 < /dev/null &
tail -f runs/synthesis/logs/driver.log
```

依次执行：Qwen 0.5B 弱回答 → 原版 LLaMA 8B 强回答 → LLaMA 对照两者生成批判 → 导出 Alpaca 格式监督样本。按固定输入、批次和种子记录断点；在配置不变时重新运行 `launch.py` 可继续已保存的合成进度。

```bash
python3 runs/synthesis/run.py status
```

出现 `CLM_CFT_SYNTHESIS_COMPLETE` 且状态为 `complete` 后再继续。主要输出：

```text
runs/synthesis/artifacts/full/cft_train_10k.json
runs/synthesis/artifacts/full/quality_summary.json
runs/synthesis/artifacts/full/review_preview70.json
```

`quality_summary.json` 报告长度截断等生成情况；沿用历史未筛选训练流程，未宣称自动通过语义质量审核。不要在任务尚未完成时开始 SFT。

后台命令中的 `tail -f` 仅查看日志，按 Ctrl+C 退出查看不会终止 `nohup` 启动的任务。合成、SFT、GRPO 和评测按顺序运行，避免同时占用四张 GPU。

## 5. 训练 SFT CLM8B 并合并权重

```bash
python3 tools/prepare_workdir.py sft --config configs/paths.json \
  --corpus runs/synthesis/artifacts/full/cft_train_10k.json \
  --accept-new-corpus --output runs/sft_train

nohup python3 -u tools/launch.py --workdir runs/sft_train \
  > runs/sft_train/logs/driver.log 2>&1 < /dev/null &
tail -f runs/sft_train/logs/driver.log
```

自动编码、随机划分 9,000 train / 1,000 validation、先跑 3 步短测，再正式训练两轮。LoRA rank16、alpha64、学习率 1e-5，四卡每卡 batch1、梯度累积8。只监督目标输出；编码若超出 6,144 token 或掩码不符会报错，不会偷偷截断。需要检查新生成语料后另行调整配置，不能当作训练已完成。

查看进度：

```bash
"$TRAIN_PY" runs/sft_train/run.py status
```

出现 `CLM_LORA_TRAINING_COMPLETE` 且状态 `complete/full_complete` 后，使用同一训练环境合并。配置默认已指向本次生成的语料和适配器：

```bash
"$TRAIN_PY" tools/merge_sft.py --config configs/paths.json \
  --adapter runs/sft_train/artifacts/full \
  --corpus runs/synthesis/artifacts/full/cft_train_10k.json \
  --output runs/models/CLM8B-SFT
```

输出完整 BF16 模型、tokenizer 和 `merge_manifest.json`。合并检查参数有限、抽样层发生变化，并保存模型来源及文件哈希。只想做 SFT 组可以跳过下一节 GRPO。

## 6. 在自己的 SFT CLM8B 上训练 GRPO

```bash
python3 tools/prepare_workdir.py grpo --config configs/paths.json \
  --data-dir data/prepared/grpo --accept-new-data --output runs/grpo

nohup python3 -u tools/launch.py --workdir runs/grpo \
  > runs/grpo/logs/pipeline.log 2>&1 < /dev/null &
tail -f runs/grpo/logs/pipeline.log
```

程序先校验环境、奖励判别规则和模型来源，再完成初始回答、开发集前测、16 轮候选批判采样/回答修订/奖励评分/LoRA 更新、开发集后测。Qwen 全程冻结；只训练 SFT 基座上新增的 rank8、alpha16 RL LoRA。每题4个候选、每轮8题；无法判定的奖励组不用于更新，零优势组不贡献梯度，存在质量停止检查。因此“16轮”不保证每轮都有有效 optimizer step。

```bash
"$TRAIN_PY" runs/grpo/run.py status
```

状态 `complete` 后查看 `runs/grpo/reports/summary.json`。最终产物：

```text
runs/grpo/artifacts/checkpoints/step016/adapter_model.safetensors
runs/grpo/artifacts/checkpoints/step016/adapter_config.json
runs/grpo/artifacts/spec.json
```

默认 `rl_adapter` 已指向上述目录。评测时加载**这次 SFT 合并模型 + 这次 RL adapter**，并核对父模型哈希。不能把 RL adapter 挂到原版 LLaMA 或另一轮 SFT 上。同一配置重新运行 `launch.py` 可恢复已校验完成的 GRPO 阶段；若奖励校准或质量检查失败，应查看阶段日志，不要跳过检查来宣称训练完成。

## 7. 四组评测与 Kaggle CSV

先保证之前的训练进程结束、四卡空闲。`--mode smoke` 只跑5条，不能提交 Kaggle；`full` 才是完整 12,576 条。所有输出目录必须是新目录。

先生成本次 Qwen 基线，后三组共享这一份初始回答（默认 `baseline` 配置已指向该位置）：

```bash
python3 tools/evaluate.py --config configs/paths.json \
  --group qwen --mode smoke --output runs/qwen_smoke

python3 tools/evaluate.py --config configs/paths.json \
  --group qwen --mode full --output runs/qwen_full
```

后三组分别做短测、全量，顺序运行，不要同时抢占四张 GPU：

```bash
python3 tools/evaluate.py --group frozen --mode smoke --output runs/frozen_smoke
python3 tools/evaluate.py --group frozen --mode full --output runs/frozen_full

python3 tools/evaluate.py --group sft --mode smoke --output runs/sft_smoke
python3 tools/evaluate.py --group sft --mode full --output runs/sft_full

python3 tools/evaluate.py --group rl --mode smoke --output runs/rl_smoke
python3 tools/evaluate.py --group rl --mode full --output runs/rl_full
```

以上默认读取 `configs/paths.json`。如果改过 Qwen 全量输出目录，修改 `baseline`，或后三组都显式传相同的 `--baseline /path/to/result_rationale_cot.json`。不要混用不同批次的初始回答。

全量推理耗时较长，可按以下模式后台运行其中一个任务（不要与同一任务的前台命令重复启动）：

```bash
mkdir -p runs/driver_logs
nohup python3 -u tools/evaluate.py --group rl --mode full --output runs/rl_full \
  > runs/driver_logs/rl_full.log 2>&1 < /dev/null &
tail -f runs/driver_logs/rl_full.log
# 逐题进度在 runs/rl_full/phase2.log 或 phase3.log
```

日志出现 `EVALUATION_COMPLETE` 且各输出目录的 `run_manifest.json` 为 `status=complete` 后，提交对应文件：

| 实验 | Kaggle 提交文件 |
| --- | --- |
| Qwen | `runs/qwen_full/submission_qwen_full_rationale.csv` |
| Frozen CLM | `runs/frozen_full/submission_frozen_full_rationale.csv` |
| SFT CLM | `runs/sft_full/submission_sft_full_rationale.csv` |
| SFT + GRPO | `runs/rl_full/submission_rl_full_rationale.csv` |

CSV 固定列名 `id,answer`，`answer` 是 Qwen 完整最终回答，不做短答案提取，不把有疑虑的回答替换为 unknown。导出校验模板顺序、题数、非空回答及 CSV 读回一致性，并生成 `.manifest.json`。多行回答会占多行文本，不能用 `wc -l` 统计题数。代码不自动向 Kaggle 上传。

已有完整 JSON 时，可独立导出一个新文件，不需加载模型：

```bash
python3 tools/export_submission.py \
  --rationale runs/rl_full/2WikiMultiHopQA/result_refine_cot.json \
  --sample data/prepared/course/sample_submission.csv \
  --output runs/rl_full/submission_copy.csv
```

## 8. 检查、结果与历史备份

```bash
python3 tools/check_release.py
python3 -m unittest discover -s tests -v
```

当前新入口通过 CPU 测试及本地真实数据准备验证；没有在本机 Windows 上重新执行 GPU 训练、合并或推理。新增验证记录见 [从头运行验证](provenance/from_scratch_validation.json)。历史四组已在服务器完成；便携入口的新运行仍需在读者 GPU 环境验证。不同模型 revision、生成采样、批次、浮点计算和引擎配置都会影响结果。

- [实现与实验报告](docs/实现与实验报告.md)：同学写报告/PPT用的算法、实现和历史实验说明。
- [报告与PPT写作提纲](docs/报告与PPT写作提纲.md)：报告章节、幻灯片提纲。
- [代码导航与复现范围](docs/代码导航与复现范围.md)：新入口与历史代码对应关系。
- [历史服务器备份说明](docs/服务器文件下载与部署.md)：仅供原实验归档/恢复；私有包不是本 README 的前置条件。

训练与评测使用数据中已有的检索文档，没有新增在线检索。SFT 的 ASQA 部分支持口径、原论文与课程实验差异、GRPO 小规模训练和推理引擎差异均见详细报告。

上游：[RAG-ReasonAlignment](https://github.com/upup-wei/RAG-ReasonAlignment)，历史代码提交 `11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a`。来源和授权范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。基础模型、作者数据和课程数据按各自来源条件使用。
