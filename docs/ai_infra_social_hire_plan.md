# 在职社招转 AI Infra 实操版

> 这版不是“方向建议”，而是“你到底要做出什么”。
>
> 适用前提：
> - 你在职
> - 每周稳定投入 `8-12` 小时就不错了
> - 目标不是成为最强 CUDA 选手，而是做出一个能拿去社招讲的 AI infra 项目

## 1. 先说结论

如果这份文档读完以后，你还不知道自己要交付什么，那这份文档就是失败的。

所以这里先把最终目标写死：

## 你的项目最终应该长成这样

8 周后，至少要有下面 `4` 个可展示成果：

1. `1` 篇 kernel 深读文档
   你能讲清一个具体 kernel 怎么并行、怎么搬数据、瓶颈在哪。

2. `1` 套 benchmark 结果
   不是口头说“更快”，而是有表格、有图、有结论。

3. `1` 个小型 runtime / dispatch 脚本
   输入 workload，输出选择哪条路径，或者实际跑对应实现。

4. `1` 篇项目总结
   能给面试官讲：问题是什么、你做了什么、结果是什么、下一步是什么。

如果你最后只有“我看了很多代码”，那这个项目不算完成。

---

## 2. 这个项目不是什么

为了防止你做散，先把非目标写清楚。

## 这次不追求

- 不追求自己发明新注意力算法
- 不追求重写整个 cuLA
- 不追求一口气精通所有 kernel
- 不追求做完整生产级 serving 系统
- 不追求把训练、推理、编译器、服务全都做一遍

## 这次只追求

- 吃透 `1-2` 个有代表性的 kernel
- 跑出一套像样的 benchmark
- 做一个小而完整的 dispatch / runner
- 能把这件事讲成 AI infra 项目，而不是学习笔记

---

## 3. 最小可交付版本

如果时间真的紧，项目最小闭环就是这四样。

## Deliverable A: Kernel Note

二选一即可：

- `la_decode`
- `fwd_o`

最好做到：

- 你能画 CTA / warp 分工图
- 你能讲清 GMEM / SMEM / RMEM / TMEM 的流向
- 你能说出为什么这么设计

最终产物建议是一个 markdown 文档，比如：

- `docs/kernel_note_la_decode.md`
- `docs/kernel_note_fwd_o.md`

## Deliverable B: Benchmark Report

至少覆盖一类 workload：

- decode
- 或 prefill / chunked recurrent

最终产物建议是：

- `benchmarks/results/*.csv`
- `benchmarks/results/*.md`
- `benchmarks/results/*.png`

至少包含：

- 配置
- latency
- throughput
- 简短结论

## Deliverable C: Dispatch / Runner

最简单也可以只是一个脚本：

- 输入：`mode=batch/seq_len/varlen/device`
- 输出：推荐路径或直接调用对应 benchmark / kernel wrapper

最终产物建议是：

- `tools/run_workload_matrix.py`
- 或 `tools/dispatch_demo.py`

## Deliverable D: Project Summary

最终要有一个总说明，给面试和 GitHub 用。

最终产物建议是：

- `docs/project_summary.md`

结构固定成：

1. 我想解决什么问题
2. 我选了哪条技术路线
3. 我做了哪些工程工作
4. 结果怎么样
5. 如果继续做，下一步是什么

---

## 4. 这 8 周你到底做什么

下面不是“建议”，而是推荐默认路线。

## Week 1: 仓库地图

### 本周目标

回答一句话：

`cuLA` 里哪些是 kernel，哪些是调度，哪些是 benchmark，哪些是 C++ 扩展。

### 本周只看这些文件

- `README.md`
- `REPO_LAYOUT.md`
- `setup.py`
- `docs/cuda_learning_guide.md`
- `cula/lightning/la_decode.py`

### 本周必须产出

写一页自己的仓库地图，哪怕只是本地笔记，也至少要回答：

- 项目主语言和 DSL 是什么
- Blackwell / Hopper 分别走哪条路
- `la_decode`、`fwd_o`、`chunk_delta_h` 分别负责什么
- Python、CuTe DSL、CUDA C++ 三层怎么衔接

### 本周完成标准

你能不看文档，口头讲出：

“这个仓库里，推理相关的 kernel 主线是怎么分层的。”

如果做不到，就别急着往下走。

---

## Week 2: 吃透一个 kernel

### 本周目标

把一个 kernel 讲清楚，不是“看过”，而是“能复述”。

### 默认选择

先读：

- `cula/lightning/la_decode.py`

如果时间够，再看：

- `docs/lightning_attn_pipeline.md`

### 本周必须产出

写一篇短文档，建议文件名：

- `docs/kernel_note_la_decode.md`

文档里必须回答这 `6` 个问题：

1. 一个 CTA 负责什么
2. 一个 warp 负责什么
3. 为什么要有 staged shared memory
4. `cp.async` 在这里干什么
5. 数据如何从输入变成输出
6. 这个 kernel 的潜在瓶颈是什么

### 本周完成标准

你能用 `5` 分钟白话讲清这个 kernel。

如果你自己讲着讲着就卡住，说明还没懂。

---

## Week 3: 第一次 benchmark

### 本周目标

从“看代码”切到“拿数据说话”。

### 本周默认做法

从这几个里选一个最容易跑通的：

- `benchmarks/bench_la_decode_vs_fla.py`
- `benchmarks/bench_fwd_o.py`
- `benchmarks/bench_chunk_delta_h.py`

### 本周必须产出

至少做出下面三样：

1. 一份原始结果表
2. 一张图
3. 三条结论

三条结论至少要包含：

- 哪种 workload 下收益明显
- 哪种 workload 下收益一般
- 你对瓶颈的一个猜测

### 本周完成标准

你能把 benchmark 结果讲成一句完整的话，而不是只报数字。

错误示范：

- “这里是 1.23x、1.45x、1.18x”

正确示范：

- “收益主要出现在较大 batch 或较长 seq 的配置，小 workload 下 launch 和数据搬运成本占比更高。”

---

## Week 4: 再吃透一个更复杂 kernel

### 本周目标

再补一个更能体现 CUDA 工程深度的 kernel。

### 二选一

- `cula/ops/fwd_o.py`
- `cula/ops/chunk_delta_h.py`

如果你更想学：

- pipeline / dual-ACC / TMEM：选 `fwd_o`
- register-carry / persistent scheduling：选 `chunk_delta_h`

### 本周必须产出

再写一篇短文档，建议文件名：

- `docs/kernel_note_fwd_o.md`
- 或 `docs/kernel_note_chunk_delta_h.md`

这篇文档必须额外回答：

- 为什么这个 kernel 比 `la_decode` 更复杂
- 复杂度主要来自哪里
- 这个设计在服务长上下文 / recurrent update 上有什么价值

### 本周完成标准

你已经不是“看懂一个小 kernel”，而是能对比两个 kernel 的设计差异。

---

## Week 5: 做一个小工具

### 本周目标

把零散 benchmark 和 workload 配置组织成一个小工具。

### 默认做法

做一个 runner，功能尽量小：

- 输入 workload 配置
- 选择跑哪个 benchmark
- 输出统一格式的结果

例如支持这些参数：

- `--mode decode|prefill`
- `--batch-size`
- `--seq-len`
- `--varlen`
- `--out results.csv`

### 本周必须产出

至少新增一个脚本，建议文件名：

- `tools/run_workload_matrix.py`

脚本不需要很聪明，但必须满足：

- 参数清晰
- 输出可复用
- 能批量跑几组 case

### 本周完成标准

你开始有一点 runtime / dispatch 的雏形，而不是只能手动跑脚本。

---

## Week 6: 把结果整理成“项目”

### 本周目标

把前面做的东西从“零散记录”变成“一个能展示的项目”。

### 本周必须产出

至少整理出：

1. 一个总 README 或 summary
2. benchmark 图表
3. kernel note 索引
4. 你的小工具使用方式

推荐写：

- `docs/project_summary.md`

里面固定写四段：

1. 背景：为什么关注线性注意力推理
2. 方法：我具体看了哪些 kernel，做了哪些 benchmark，写了什么工具
3. 结果：哪些场景收益明显，哪些不明显
4. 结论：这件事对 AI infra 有什么价值

### 本周完成标准

不懂这个仓库的人看你的 summary，也能知道你到底做了什么。

---

## Week 7: 补一块调度视角

### 本周目标

让项目从“懂 kernel”升级成“懂推理链路”。

### 本周只做一件事

把你的 benchmark / runner 和调度视角连起来。

最简单可以只写一篇文档：

- `docs/scheduling_note.md`

里面回答：

- prefill 和 decode 为什么要分开看
- varlen workload 为什么会影响调度
- 为什么底层 kernel 特性会影响 dispatch 决策
- 如果以后做 continuous batching，这个项目哪些结论还能复用

### 本周完成标准

你能把项目讲成：

“我不是单纯在看 kernel，我是在看 workload、kernel 和 runtime 决策的关系。”

这一步对社招很重要。

---

## Week 8: 面试化

### 本周目标

把技术成果转换成面试表达。

### 本周必须产出

至少写好这三样：

1. `3` 条简历 bullet
2. `1` 分钟项目介绍
3. `5` 分钟技术讲解稿

### 简历 bullet 至少要长这样

- 阅读并分析基于 CuTe DSL 的线性注意力推理 kernel，梳理 CTA、warp specialization、memory hierarchy 与 pipeline 设计
- 构建 benchmark 与 workload runner，对 decode / prefill 场景在不同 batch、seq length、varlen 配置下的性能进行评测与对比
- 从 runtime dispatch 视角总结 kernel 特性与 workload 形态之间的关系，形成项目技术文档与可复现结果

### 本周完成标准

你已经可以拿这个项目去面试了。

---

## 5. 如果你每周只有 5 小时

那就做缩减版，不要硬追 8 周完整版。

## 保留

- `1` 个 kernel note
- `1` 套 benchmark
- `1` 个 runner
- `1` 份 project summary

## 砍掉

- 第二个复杂 kernel
- 额外 serving demo
- 过深的 C++ fused kernel 阅读

对在职社招来说，**小而完整** 比 **大而空** 更有价值。

---

## 6. 每周真正该问自己的，不是“学了多少”

而是下面这三个问题：

1. 我这周留下了什么可展示产物
2. 我这周新增了什么可以讲给面试官听的内容
3. 我这周做的事情，和 AI infra 的关系是什么

如果三问都答不上来，就说明又开始发散了。

---

## 7. 你现在就可以开始的第一步

今天不要做太多，只做这三件事：

1. 读 `cula/lightning/la_decode.py`
2. 用自己的话写下：
   - 这个 kernel 算什么
   - 一个 CTA 在做什么
   - 为什么需要 shared memory 和 pipeline
3. 新建一个草稿文档：
   - `docs/kernel_note_la_decode.md`

如果今天你能把这三件事做完，这个项目就从“方向焦虑”正式进入“可执行状态”了。

---

## 8. 最后一句

这份计划真正想逼你做的，不是“继续想”，而是每两周都沉淀一个具体文件、一个具体脚本、或者一份具体结果。

只要仓库里开始稳定出现这些真实产物，这条路就是有效的。
