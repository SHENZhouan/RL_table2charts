# Plotly Finetuned 0420 Pipeline 说明

本文档记录本轮 `plotly_finetuned_0420` 实验的完整流程，方便后续复用、续跑和排查。实验内容为：

```text
Plotly full dataset -> SFT -> RL -> final evaluation
```

本轮最终产物已创建易识别别名：

```text
Results/Models/plotly_finetuned_0420_sft -> 20260420152824-2el192fd128.128GRUh-allCharts
Results/Models/plotly_finetuned_0420_rl  -> 20260421001338-2el192fd128.128GRUh-allCharts-RL
```

## 环境

代码根目录：

```bash
/ssd/shenzhouan/Table2Charts
```

训练代码目录：

```bash
/ssd/shenzhouan/Table2Charts/Table2Charts
```

Python 环境：

```bash
/ssd/shenzhouan/Table2Charts/.venv/bin/python
```

数据集：

```bash
Data/PlotlyTable2Charts
```

本轮使用的主要配置：

```text
model_name       = cp
model_size       = small
features         = all-fast
search_type      = allCharts
input_type       = allCharts
previous_type    = allCharts
lang             = en
SFT epochs       = 1
RL epochs        = 1
search_limits    = e50-b4-na
GPU              = 3,4,5,6
```

## 持久运行方式

长实验应放进 `tmux`，这样断开 SSH 或关闭本地电脑后实验仍会继续跑。

启动完整 pipeline：

```bash
cd /ssd/shenzhouan/Table2Charts
tmux new-session -d -s plotly_small_pipeline \
  "cd /ssd/shenzhouan/Table2Charts && \
   GPU_IDS=3,4,5,6 \
   SFT_NPROCS=4 \
   EVAL_NPROCS=4 \
   RUN_SFT_EVAL=0 \
   MASTER_PORT=29616 \
   Results/run_logs/run_plotly_small_tmux_pipeline.sh"
```

查看会话：

```bash
tmux ls
tmux attach -t plotly_small_pipeline
```

查看日志和状态：

```bash
tail -f /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_small_tmux_pipeline.latest.log
cat /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_small_tmux_pipeline.latest.status
nvidia-smi
```

## 完整 Pipeline 脚本

完整脚本路径：

```text
Results/run_logs/run_plotly_small_tmux_pipeline.sh
```

脚本做了这些事：

1. 生成 `RUN_ID`
2. 写入 log/status 文件
3. 跑 SFT
4. 可选跑 SFT evaluation
5. 跑 RL
6. 自动找到 RL checkpoint
7. 跑最终 RL evaluation
8. 写入 `status=finished`

本轮为了减少耗时设置了：

```bash
RUN_SFT_EVAL=0
```

因此 SFT 完成后跳过中间 SFT eval，直接进入 RL。

## Step 1: SFT

本轮 SFT 使用的命令形状如下：

```bash
cd /ssd/shenzhouan/Table2Charts/Table2Charts

CUDA_VISIBLE_DEVICES=3,4,5,6 \
MASTER_ADDR=localhost \
MASTER_PORT=29616 \
/ssd/shenzhouan/Table2Charts/.venv/bin/python -m nn_train.pretrain \
  --model_name=cp \
  --model_size=small \
  --features=all-fast \
  --train_batch_size=64 \
  --valid_batch_size=64 \
  --log_freq=200 \
  --negative_weight=0.8 \
  --search_type=allCharts \
  --input_type=allCharts \
  --previous_type=allCharts \
  --model_save_path=../Results/Models \
  --summary_path=../Results/summary \
  --corpus_path=../Data/PlotlyTable2Charts \
  --lang=en \
  --epochs=1 \
  --nprocs=4 \
  --num_workers=2
```

参数含义：

```text
CUDA_VISIBLE_DEVICES=3,4,5,6
    只暴露 3/4/5/6 四张 GPU 给本实验，避免占满机器。

MASTER_ADDR / MASTER_PORT
    PyTorch distributed 进程组通信地址和端口。

-m nn_train.pretrain
    运行监督微调，也就是 SFT。

--model_name=cp
    使用 CopyNet 模型。

--model_size=small
    使用 small 模型配置。

--features=all-fast
    使用 fasttext 语义特征以及完整结构/类型/数据特征。

--train_batch_size=64
--valid_batch_size=64
    每张 GPU 的 batch size。

--negative_weight=0.8
    NLLLoss 中负类权重。

--search_type=allCharts
--input_type=allCharts
--previous_type=allCharts
    训练/输入/action space 都使用 line/bar/scatter/pie 等 chart 类型集合。

--model_save_path=../Results/Models
    checkpoint 输出目录。

--summary_path=../Results/summary
    TensorBoard event 输出目录。

--corpus_path=../Data/PlotlyTable2Charts
    使用 Plotly full corpus。

--epochs=1
    SFT 训练 1 个 epoch。

--nprocs=4
    启动 4 个分布式训练进程。

--num_workers=2
    每个 DataLoader 使用 2 个 worker。
```

SFT 产物：

```text
Results/Models/20260420152824-2el192fd128.128GRUh-allCharts/states_ep0.pt
Results/Models/plotly_finetuned_0420_sft/states_ep0.pt
```

别名路径推荐后续使用：

```text
Results/Models/plotly_finetuned_0420_sft/states_ep0.pt
```

## Step 2: 可选 SFT Evaluation

本轮没有跑 SFT eval，以减少总耗时。

如果未来需要对 SFT checkpoint 单独评估，可以设置：

```bash
RUN_SFT_EVAL=1
```

或者单独运行：

```bash
cd /ssd/shenzhouan/Table2Charts/Table2Charts

CUDA_VISIBLE_DEVICES=3,4,5,6 \
/ssd/shenzhouan/Table2Charts/.venv/bin/python test_agent_mp.py \
  -m ../Results/Models/plotly_finetuned_0420_sft \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/test-sft-plotly-small-manual \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 4 \
  --nagents 64 \
  --nthreads 5 \
  --search_limits e50-b4-na \
  --corpus_path ../Data/PlotlyTable2Charts \
  --lang en \
  --limit_search_group
```

## Step 3: RL

RL 从 SFT checkpoint 初始化。

本轮曾在完整 pipeline 的 RL 启动处遇到 PyTorch launcher 参数兼容问题：新版 launcher 会传 `--local-rank`，原代码只接受 `--local_rank`。已修复：

```python
parser.add_argument("--local_rank", "--local-rank", ...)
```

此外，为让 `queue_mode=local` 支持多卡，本轮将 local tUID 列表按 rank 分片：

```python
train_tuids[global_rank::world_size]
valid_tuids[global_rank::world_size]
```

这样 RL 可以 4 卡并行，并仍覆盖全量 train/valid 表。

RL 命令形状如下：

```bash
cd /ssd/shenzhouan/Table2Charts/Table2Charts

CUDA_VISIBLE_DEVICES=3,4,5,6 \
/ssd/shenzhouan/Table2Charts/.venv/bin/python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29617 \
  script.py \
  --corpus_path=../Data/PlotlyTable2Charts \
  --model_size=small \
  --model_name=cp \
  --features=all-fast \
  --negative_weight=0.8 \
  --search_limits=e50-b4-na \
  --epochs=1 \
  -m ../Results/Models \
  -p /ssd/shenzhouan/Table2Charts/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt \
  --summary_path=../Results/summary \
  --search_type=allCharts \
  --input_type=allCharts \
  --previous_type=allCharts \
  --lang=en \
  --queue_mode=local \
  --log_freq_agent=500 \
  --log_freq_batch=100 \
  --max_tables=64 \
  --min_memory=1000 \
  --memory_sample_size=64 \
  --memory_sample_rounds=2
```

参数含义：

```text
-m ../Results/Models
    RL 模型输出目录。

-p <SFT checkpoint>
    从 SFT checkpoint 初始化 RL。

--queue_mode=local
    不依赖 RabbitMQ，直接在本地进程中分发表。

--search_limits=e50-b4-na
    搜索预算：expand_limit=50, beam_size=4, 不使用 r4c2 限制。

--max_tables=64
    每个 student 同时维护的 table agent 上限。

--min_memory=1000
    replay memory 达到 1000 后开始学习。

--memory_sample_size=64
    每轮从 replay memory 采样的 batch size。

--memory_sample_rounds=2
    每次 agent 扩展后做 2 轮 replay learning。

--log_freq_agent=500
    每完成约 500 个 agents 记录一次搜索进度。

--log_freq_batch=100
    每 100 个 replay batch 记录一次训练指标。
```

RL 产物：

```text
Results/Models/20260421001338-2el192fd128.128GRUh-allCharts-RL/states_ep0.pt
Results/Models/plotly_finetuned_0420_rl/states_ep0.pt
```

别名路径推荐后续使用：

```text
Results/Models/plotly_finetuned_0420_rl/states_ep0.pt
```

RL 训练日志：

```text
Results/Models/plotly_finetuned_0420_rl/evaluations/train/
Results/Models/plotly_finetuned_0420_rl/evaluations/test-valid/
```

## Step 4: Final Evaluation

最终 evaluation 使用的是 Plotly test split，不是 WebTable。

命令形状如下：

```bash
cd /ssd/shenzhouan/Table2Charts/Table2Charts

CUDA_VISIBLE_DEVICES=3,4,5,6 \
/ssd/shenzhouan/Table2Charts/.venv/bin/python test_agent_mp.py \
  -m ../Results/Models/plotly_finetuned_0420_rl \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/test-rl-plotly-small-20260420T161243Z \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 4 \
  --nagents 64 \
  --nthreads 5 \
  --search_limits e50-b4-na \
  --corpus_path ../Data/PlotlyTable2Charts \
  --lang en \
  --limit_search_group
```

参数含义：

```text
-m ../Results/Models/plotly_finetuned_0420_rl
    载入 RL 权重目录。

-f states_ep0.pt
    载入第 0 个 epoch 的 checkpoint。

--log_save_path evaluations/test-rl-plotly-small-20260420T161243Z
    evaluation 输出子目录，实际位于模型目录下。

--nprocs 4
    启动 4 个搜索进程。

--nagents 64
    每个进程并行处理的 agent/table 数。

--nthreads 5
    每个进程内部用于 agent 调度的线程数。

--limit_search_group
    Plotly 数据没有 Group，因此不搜索 Group。

--use_valid_set 未设置
    默认使用 test split。

--web_table 未设置
    不使用 WebTable 模式。
```

最终 evaluation 产物：

```text
Results/Models/plotly_finetuned_0420_rl/evaluations/test-rl-plotly-small-20260420T161243Z/[test-summary]20260421T0317.log
```

本轮最终结果：

```text
t_cnt      = 12946
R@1        = 0.4178124517
R@3        = 0.7735980226
R@5        = 0.8326896339
R@10       = 0.9022091766
R@20       = 0.9528811988
all        = 0.9533446624
first_rank = 2.91 * 12342
```

## Resume: 只从 SFT 继续跑 RL + Evaluation

如果 SFT 已经完成，后续只改 reward、RL 参数或 evaluation 参数，通常不需要重新 SFT。可以直接从 SFT checkpoint 续跑：

```bash
cd /ssd/shenzhouan/Table2Charts

tmux new-session -d -s plotly_small_resume_rl_eval \
  "cd /ssd/shenzhouan/Table2Charts && \
   GPU_IDS=3,4,5,6 \
   RL_NPROCS=4 \
   EVAL_NPROCS=4 \
   MASTER_PORT=29617 \
   SFT_CKPT=/ssd/shenzhouan/Table2Charts/Results/Models/plotly_finetuned_0420_sft/states_ep0.pt \
   Results/run_logs/run_plotly_small_resume_rl_eval.sh"
```

Resume 脚本路径：

```text
Results/run_logs/run_plotly_small_resume_rl_eval.sh
```

状态与日志：

```bash
tail -f /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_small_resume_rl_eval.latest.log
cat /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_small_resume_rl_eval.latest.status
```

该脚本会自动执行：

```text
SFT checkpoint -> RL -> final RL evaluation
```

## 何时需要重新 SFT

一般不需要重新 SFT 的情况：

```text
只改 reward 函数
只改 RL search reward shaping
只改 RL replay memory 参数
只改 RL epoch 数
只改 final evaluation 参数
```

建议重新 SFT 的情况：

```text
换数据集
改 model_size 或模型结构
改 features，例如从 all-fast 换到其他配置
改 search_type/input_type/previous_type
改 action space 或 token 表达
改 supervised target/Q-value 构造逻辑
```

## WebTable Evaluation 说明

代码支持：

```bash
--web_table
```

但当前仓库里 `Data/WebTable/` 只有 WebTable 处理代码，没有已经转成 Table2Charts 格式的 WebTable corpus。缺少：

```text
index/schema_ids.json
data/*.t0.DF.json
embeddings/fasttext/*.EMB.json
sample-new/*.sample.json
```

因此目前不能直接跑 WebTable evaluation。准备好 WebTable corpus 后，命令形状为：

```bash
cd /ssd/shenzhouan/Table2Charts/Table2Charts

CUDA_VISIBLE_DEVICES=3,4,5,6 \
/ssd/shenzhouan/Table2Charts/.venv/bin/python test_agent_mp.py \
  -m ../Results/Models/plotly_finetuned_0420_rl \
  -f states_ep0.pt \
  --model_name cp \
  --model_size small \
  --features all-fast \
  --log_save_path evaluations/webtable-plotly-finetuned-0420 \
  --search_type allCharts \
  --input_type allCharts \
  --previous_type allCharts \
  --nprocs 4 \
  --nagents 64 \
  --nthreads 5 \
  --search_limits e50-b4-na \
  --corpus_path <WEBTABLE_TABLE2CHARTS_CORPUS_PATH> \
  --lang en \
  --web_table \
  --empirical_log_path ../Results/WebTable/plotly_finetuned_0420
```

WebTable 模式默认没有 ground truth chart，因此不会产生 Plotly test 那样的 `R@1/R@3/R@5` recall 指标，主要用于生成和记录推荐结果。

## 常见问题

### 1. PyTorch launcher 报 `unrecognized arguments: --local-rank`

原因：新版 `torch.distributed.launch` 会传 `--local-rank`，旧代码只接受 `--local_rank`。

修复方式：

```python
parser.add_argument("--local_rank", "--local-rank", default=0, type=int, metavar='N', ...)
```

### 2. RL local 多卡会重复处理数据吗

本轮已改为按 rank 分片：

```python
train_tuids[global_rank::world_size]
valid_tuids[global_rank::world_size]
```

因此 local queue 多卡会覆盖全量数据，但每个 rank 处理不同切片。

### 3. `nvidia-smi` 显示 GPU util 低是不是卡住了

不一定。RL 和 evaluation 的主要耗时在搜索/agent 调度/CPU 逻辑，GPU 只负责模型打分。GPU util 低但进程、日志和 agent 进度正常，通常说明程序仍在跑。

### 4. 如何确认最终完成

看 status 文件：

```bash
cat /ssd/shenzhouan/Table2Charts/Results/run_logs/plotly_small_resume_rl_eval.latest.status
```

完成时应有：

```text
status=finished
finished_utc=...
```

或者看 final evaluation 日志末尾：

```text
Test finished!
Complete recall info: ...
```
最终 evaluation 关键结果：

t_cnt = 12946
R@1  = 0.4178124517
R@3  = 0.7735980226
R@5  = 0.8326896339
R@10 = 0.9022091766
R@20 = 0.9528811988
all  = 0.9533446624
first_rank = 2.91 * 12342
RL 训练阶段也正常保存了模型，训练 summary 里最终大致是：

precision = 0.541323
recall    = 0.358383
f1        = 0.431254
R@1       = 0.2073098969
R@3       = 0.3956149336
R@5       = 0.5014711331
R@10      = 0.6527036157