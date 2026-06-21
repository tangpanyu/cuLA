# cuLA CUDA 学习指南

> 目标：帮助你从 **CUDA 内核设计** 的角度理解这个仓库，而不只是“知道有哪些文件”。
>
> 建议阅读对象：
> - 已经会基本的 PyTorch / GPU 编程
> - 对 CUDA 的 thread block、warp、SMEM 有初步概念
> - 想借这个仓库学习现代 NVIDIA GPU 上的高性能 kernel 设计

## 1. 先用一句话理解这个仓库

`cuLA` 是一个面向 **线性注意力 / delta-rule / Lightning Attention** 的 GPU kernel 仓库，主实现路线是：

- Python 入口与调度
- **CuTe DSL** 写的 CUDA kernel
- 部分 Triton baseline / 对照实现
- 部分 CUTLASS C++ / CUDA 扩展

如果你从 CUDA 角度读它，最重要的不是先记住算法名字，而是先抓住这三件事：

1. 这个 kernel 的 **并行单位** 是什么
2. 数据在 **GMEM / SMEM / TMEM / RMEM** 之间怎么流
3. 计算和搬运是怎么通过 **pipeline / barrier / persistent scheduling** 重叠起来的

---

## 2. 这个仓库里“CUDA”具体分几层

先建立地图，再进细节。

### 2.1 Python 调度层

这些文件负责参数整理、shape 检查、调度不同实现、对接测试和 benchmark：

- `cula/kda/chunk.py`
- `cula/kda/chunk_fwd.py`
- `cula/kda/__init__.py`
- `cula/lightning/__init__.py`

这一层要回答的问题：

- 用户调用哪个 API
- 最后落到哪个 kernel
- 是 Blackwell 还是 Hopper 路径
- 是 fixed-length 还是 varlen

### 2.2 CuTe DSL kernel 层

这是你学习 CUDA 的重点区：

- `cula/ops/fwd_o.py`
- `cula/ops/chunk_delta_h.py`
- `cula/ops/lightning_attn.py`
- `cula/lightning/la_decode.py`

这一层虽然写在 Python 里，但本质是在写 GPU kernel。你需要把它当成“CUDA kernel 的 DSL 写法”来读。

重点观察：

- CTA / warp 分工
- tile shape
- TMA / cp.async
- SMEM / TMEM / RMEM
- named barrier / pipeline stage
- 持久化 kernel 的 work scheduling

### 2.3 Triton 层

这部分更多是参考实现、baseline 或某些尚未迁到 CuTe DSL 的模块：

- `cula/kda/chunk_intra.py`

学习价值在于：

- 帮你对照“同一个算法，Triton 写法和 CuTe DSL 写法差别在哪”
- 帮你分辨哪些优化是算法层，哪些是架构层

### 2.4 CUDA C++ / CUTLASS 层

这部分更偏“最终 fused kernel / 底层扩展”：

- `csrc/api/pybind.cu`
- `csrc/kda/sm90/`
- `csrc/kda/sm100/`

这一层适合在你读完 CuTe DSL 后再进入。因为这里更接近原生 CUTLASS / CUDA C++，抽象更底，阅读成本也更高。

---

## 3. 学这个仓库前，你需要先有的 CUDA 心智模型

如果下面这些概念不稳，建议先补一下，再回来看仓库会轻松很多。

### 3.1 执行层级

- thread
- warp
- CTA / thread block
- SM
- grid

在这个仓库里，很多 kernel 都是 **warp-specialized** 设计，也就是：

- 某些 warp 专门负责 load
- 某些 warp 专门负责 MMA
- 某些 warp 专门负责 epilog / store

所以这里不是“所有线程干一样的活”，而是明显的 **角色分工式 CTA**。

### 3.2 内存层级

至少要能区分：

- GMEM: global memory
- SMEM: shared memory
- RMEM: register
- TMEM: tensor memory / MMA operand or accumulator related storage in Blackwell 路径中频繁出现

这个仓库的核心优化，很多都围绕一句话展开：

**让数据尽量少回 GMEM，多在寄存器、SMEM、TMEM 之间流动。**

### 3.3 同步与流水

要理解：

- `__syncthreads()` 对应的 CTA 级同步思想
- warp 内同步
- producer / consumer 模型
- 多 stage pipeline
- barrier 的“谁 arrive，谁 wait，谁消费”

这个仓库不是“写个 kernel 然后 for-loop 算完”，而是典型的 **load / compute / store overlap** 设计。

### 3.4 Tile 化思维

高性能 CUDA kernel 通常不直接按整块矩阵思考，而是按 tile 思考：

- CTA 处理哪个 tile
- warp 处理 tile 的哪一部分
- MMA 指令消费什么形状的 tile
- 数据是否已经按 MMA 友好的 layout 放好

如果你读代码时总在脑里想“整个矩阵”，很容易迷路；要强迫自己改成“这个 CTA 当前只在处理一个 work unit / tile”。

---

## 4. 推荐学习顺序

不要从最大、最复杂的 fused kernel 开始。最稳的路线如下。

### 第 0 步：先看构建和架构分发

先读：

- `README.md`
- `REPO_LAYOUT.md`
- `setup.py`
- `pyproject.toml`
- `cula/utils.py`

重点看什么：

- 支持哪些架构：SM90 / SM100 / SM103
- 哪些实现是 CuTe DSL，哪些是 CUDA C++
- 编译期和运行期如何判断架构
- 哪些 kernel 是 Blackwell only

特别是 `setup.py`，你会看到：

- 根据 GPU capability 决定是否编译对应架构
- `sm_90a`、`sm_100a`、`sm_103a` 的编译 flag
- C++ extension 与 Python 包如何拼起来

这一步的目标不是学细节，而是知道：

**这个仓库不是单一路径，而是“按架构和算子能力做 dispatch 的多后端系统”。**

### 第 1 步：从 `la_decode.py` 入门

先读：

- `cula/lightning/la_decode.py`

这是非常合适的 CUDA 入门点，因为它比 `fwd_o.py` 和 `chunk_delta_h.py` 简单很多，但已经具备核心元素：

- block / warp 分工
- staged shared memory
- `cp.async`
- 计算和加载重叠
- 从 state 更新到 output 生成

读这个文件时，强烈建议你回答下面几个问题：

1. 一个 block 负责什么数据范围
2. `NUM_THREADS = 128` 是怎么分工的
3. staged SMEM 为什么需要 `NUM_STAGES = 2`
4. 为什么要先把状态 tile prefetch 到 shared memory
5. output 写回前为什么要过共享内存

如果你能把这一个文件讲清楚，后面读更复杂 kernel 就有抓手了。

### 第 2 步：读已有 pipeline 文档，再回看代码

先读文档：

- `docs/fwd_o_pipeline.md`
- `docs/chunk_delta_h_pipeline.md`
- `docs/lightning_attn_pipeline.md`

这些文档的价值很高，因为它们已经把复杂 kernel 拆成了：

- warp 角色
- pipeline 阶段
- TMEM / SMEM 用法
- persistent / non-persistent 的差异

建议读法不是“先啃代码再看文档”，而是：

1. 先看文档里的总图
2. 带着图回到代码
3. 在代码里找到文档所说的对象

这样你不容易在一堆 `cute.copy`、`pipeline.NamedBarrier`、layout 代码里迷路。

### 第 3 步：系统读 `fwd_o.py`

主文件：

- `cula/ops/fwd_o.py`
- `docs/fwd_o_pipeline.md`

这是一个非常典型的现代高性能 CUDA kernel 学习样本。建议按下面顺序读：

#### 先读顶部注释

顶部注释已经给了你：

- 数学公式
- warp specialization
- TMEM 布局
- pipeline 结构

不要跳过。这里几乎就是作者给你的“kernel 设计文档”。

#### 再看类初始化

重点看：

- `BT / BK / BV`
- `threads_per_cta`
- 各 warp id
- `persistent` 对 stage 和 occupancy 的影响
- `num_regs_cuda` 和 `num_regs_others`

你要问自己：

- 为什么 persistent 模式把 occupancy 压到 1
- 为什么换来更多 stage、更大寄存器预算
- 为什么某些缓冲双缓冲，某些不双缓冲

这一步本质是在学习 **occupancy、register、SMEM、pipeline 深度** 之间的 trade-off。

#### 再看 `_compute_grid`

这里非常重要，因为它会告诉你：

- 普通模式下 grid 怎么映射到 `(v_tile, chunk, batch*head)`
- persistent 模式下为什么直接用 `SM_count`

你会逐渐看到一个核心思想：

**persistent kernel 不是“每个 CTA 对应一个固定 tile”，而是“每个 CTA 在 SM 上驻留，然后循环吃工作”。**

#### 再看 `_plan_tmem_offsets`

这里是理解 TMEM 布局的关键。

你要重点理解：

- 为什么有 `ACC_QH` 和 `ACC_AV` 两块独立 accumulator
- 为什么要 dual-ACC
- 为什么 `qg` 和 `am` 要作为 MMA A operand 放到 TMEM

这一块体现的是：

**为了让两段 MMA 背靠背执行，作者把 accumulator 和 A operand 的布局都提前规划好了。**

#### 最后再进 `__call__`

读主体时，建议你按“生产者-消费者”拆：

- Load warp 生产什么
- CUDA warps 等什么 barrier
- MMA warp 消费什么
- Store warp 又在等什么

不要按代码行顺序死读，而要按 pipeline 依赖关系读。

### 第 4 步：系统读 `chunk_delta_h.py`

主文件：

- `cula/ops/chunk_delta_h.py`
- `docs/chunk_delta_h_pipeline.md`

这份代码比 `fwd_o.py` 更适合学习“为什么寄存器和数据流设计比公式本身更重要”。

这个 kernel 的学习重点不是公式，而是这句话：

**`h state` 在寄存器里跨 chunk 传递，避免 GMEM roundtrip。**

这是整个设计的灵魂。

读这个文件时重点抓：

- `h` 为什么放在寄存器里
- 哪一段 MMA 只是用来产生中间量 `wh`
- `v_new` 为什么先在寄存器算，再转成 TMEM operand
- `gk decay` 为什么能和 KV MMA overlap
- persistent varlen 下为什么要自己做工作调度

如果你把 `chunk_delta_h.py` 读懂，你对“高性能 CUDA kernel 的真正瓶颈不一定是算力，而是数据往返”会有非常直观的理解。

### 第 5 步：对照 Triton 实现

读：

- `cula/kda/chunk_intra.py`

目的不是把 Triton 细节啃完，而是做对照：

- Triton 在表达 block program 上更直接
- CuTe DSL 在表达 pipeline、layout、MMA operand 管理上更细

对照时重点看：

- 两边都在做 tile 化
- 两边都在做 causal / gated / chunked 计算
- 但 CuTe DSL 更接近“显式管理硬件流水和 memory movement”

这一步会帮助你建立一个判断：

**什么时候 Triton 已经够了，什么时候必须下沉到更接近 CUTLASS / CUDA 的表达层。**

### 第 6 步：最后再看 C++ fused kernel

等你前面都熟了，再看：

- `csrc/api/pybind.cu`
- `csrc/kda/sm90/`
- `csrc/kda/sm100/`

建议顺序：

1. 先看 `csrc/api/pybind.cu`，搞清楚 Python 怎么调到 C++
2. 再看 `csrc/api/kda_sm90.cu` / `csrc/api/kda_sm100.cu`
3. 最后才深入 `csrc/kda/sm90/`、`csrc/kda/sm100/`

这时你会发现很多概念已经不陌生了：

- tile scheduler
- collective mainloop
- epilogue
- persistent scheduling
- architecture-specific MMA path

---

## 5. 读每个 CUDA kernel 时的固定提问模板

你可以把下面这份清单当成“读 kernel checklist”。

### 5.1 这个 kernel 算什么

- 数学公式是什么
- 是 forward 还是 decode / prefill / recurrent update
- 是不是分成 inter-chunk / intra-chunk 两部分

### 5.2 一个 CTA 负责什么

- 一个 CTA 对应一个 chunk、一个 V tile，还是整个 sequence 的一部分
- grid 每一维映射到什么语义
- persistent 模式下 CTA 是否会循环处理多个 work unit

### 5.3 Warp 如何分工

- 哪些 warp 负责 load
- 哪些 warp 负责 MMA
- 哪些 warp 负责 epilog / store
- 是否存在“空 warp”仅用于寄存器再分配或 warp-group 要求

### 5.4 数据如何流动

- 输入先到 SMEM 还是直接进寄存器
- 哪些张量被转成 TMEM A operand
- accumulator 放在哪
- 中间结果是否回 GMEM

### 5.5 同步点在哪里

- 哪个 producer 发 barrier
- 哪个 consumer wait barrier
- barrier 保护的是“数据已到达”还是“计算已完成”

### 5.6 硬件资源怎么权衡

- 线程数多少
- 寄存器预算多少
- SMEM 占多少
- occupancy 目标是多少
- 为什么 persistent 模式和非 persistent 模式配置不同

### 5.7 性能瓶颈可能在哪

- TMA / GMEM 带宽
- register pressure
- shared memory 容量
- MMA utilization
- varlen 带来的 load imbalance

如果你每读一个文件都把这 7 个问题回答一遍，理解会非常稳。

---

## 6. 这个仓库里最值得学的 CUDA 主题

如果你的目标是“通过这个仓库补 CUDA 功底”，优先学下面这些主题。

### 6.1 Warp-specialized kernel

这里很多 kernel 不是 SIMT 风格的“所有线程做同样的事”，而是：

- load warp
- compute warp
- store warp
- epilog warp

这是现代高性能 kernel 很常见的组织方式。

你需要体会：

- 分工能减少控制流混乱
- 也会带来更复杂的同步和资源分配

### 6.2 Persistent kernel

在 `fwd_o.py` 和 `chunk_delta_h.py` 中都能看到 persistent 思路。

要重点理解：

- 为什么 grid 不再直接等于工作总数
- 为什么用 `SM_count` 或原子调度拿任务
- 为什么这能改善 load balance 和数据复用

### 6.3 Pipeline 深度设计

要关注：

- 为什么有的缓冲 1-stage
- 为什么有的缓冲 2-stage 或 3-stage
- 为什么 FP32 缓冲常常太大，不适合双缓冲

这部分非常体现工程判断力，不是教科书上几句“用双缓冲隐藏延迟”就能讲完的。

### 6.4 寄存器 carry

`chunk_delta_h.py` 是最好的例子。

很多初学者只盯着“算子公式”，但真正的高性能设计常常是：

- 中间状态不要落地
- 尽量让状态留在 RMEM
- 把一次 chunk 的输出直接变成下一次 chunk 的输入

### 6.5 架构特化

这个仓库明确区分 Hopper 和 Blackwell。

你需要建立的认识是：

- “CUDA 代码”不是对所有 GPU 一视同仁
- 不同架构支持的 MMA / TMA / TMEM 能力不同
- 高性能 kernel 往往天然是架构特化的

---

## 7. 最推荐的实战阅读路线

如果你准备花几天时间认真学，建议按这个节奏。

### Day 1: 建立地图

读：

- `README.md`
- `REPO_LAYOUT.md`
- `setup.py`
- `cula/utils.py`

目标：

- 知道仓库分层
- 知道支持哪些 GPU
- 知道 CuTe DSL 和 CUDA C++ 各在什么位置

### Day 2: 读一个相对简单的 CuTe DSL kernel

读：

- `cula/lightning/la_decode.py`

目标：

- 理解 block / warp / staged SMEM / cp.async
- 能自己画出该 kernel 的 load-compute-store 流程图

### Day 3: 读 `fwd_o`

读：

- `docs/fwd_o_pipeline.md`
- `cula/ops/fwd_o.py`
- `tests/test_fwd_o.py`
- `benchmarks/bench_fwd_o.py`

目标：

- 理解 dual-ACC
- 理解 TMEM A operand
- 理解 persistent 与 non-persistent 模式差别

### Day 4: 读 `chunk_delta_h`

读：

- `docs/chunk_delta_h_pipeline.md`
- `cula/ops/chunk_delta_h.py`
- `tests/test_chunk_delta_h.py`
- `benchmarks/bench_chunk_delta_h.py`

目标：

- 理解寄存器 carry state
- 理解 3-stage TMA pipeline
- 理解为什么 varlen persistent scheduling 更复杂

### Day 5: 读 Triton 对照实现

读：

- `cula/kda/chunk_intra.py`

目标：

- 看懂同类问题在 Triton 里如何表达
- 对比 CuTe DSL 与 Triton 的抽象层差异

### Day 6+: 再攻 fused C++

读：

- `csrc/api/pybind.cu`
- `csrc/api/kda_sm90.cu`
- `csrc/api/kda_sm100.cu`
- `csrc/kda/sm90/`
- `csrc/kda/sm100/`

目标：

- 把前面学到的 CTA / tile / pipeline / persistent 概念映射到 CUTLASS C++

---

## 8. 建议你边读边做的练习

不要只“看懂”，最好每一步都输出自己的理解。

### 练习 1：手画 work unit

对 `fwd_o.py` 和 `chunk_delta_h.py`，分别回答：

- 一个 work unit 是什么
- 一个 CTA 一次处理几个 tile
- CTA 是否跨 chunk 或跨 sequence 持续驻留

### 练习 2：手画内存流

以 `fwd_o.py` 为例，画出：

- q / g / A / h / v 从哪里来
- 哪些进 SMEM
- 哪些进 TMEM
- acc 在哪里
- o 最终怎么出去

### 练习 3：标 barrier

在代码里把每个 barrier / mbarrier / pipeline 同步点旁边写下注释：

- 谁生产
- 谁消费
- 保护的数据是什么

即使你不提交这些注释，只在本地记笔记，这个练习也很有帮助。

### 练习 4：用 benchmark 反推设计意图

读：

- `benchmarks/bench_fwd_o.py`
- `benchmarks/bench_chunk_delta_h.py`

观察：

- benchmark 如何构造输入
- 对照的是 FLA Triton 的哪一路
- 结果在 fixed-length 和 varlen 下差异如何

你的目标不是得到具体数字，而是反推：

**作者为什么要为这个场景单独写一个更复杂的 CUDA kernel。**

---

## 9. 学习时最容易卡住的点

### 9.1 把 CuTe DSL 当“普通 Python”

不要这样看。虽然语法是 Python，但这里写的是 GPU kernel 描述。

你应该把：

- `cute.Tensor`
- `cute.Layout`
- `cute.copy`
- `@cute.jit`
- `@cute.kernel`

都当成“接近 CUDA / CUTLASS 的编程对象”。

### 9.2 只看公式，不看数据流

这个仓库的性能关键大多不在公式本身，而在：

- 中间结果是否落地
- 哪些数据进入 SMEM / TMEM
- 哪些操作能 overlap
- 哪些 warp 空转、哪些 warp 饱和

### 9.3 试图一次看完整个文件

正确做法是分层：

1. 数学目标
2. CTA 和 warp 分工
3. pipeline
4. 内存布局
5. 代码细节

### 9.4 忽略 varlen

变长序列不是简单“多一个索引”而已，它会影响：

- grid 映射
- work scheduling
- load balance
- 尾块处理
- 边界检查

这个仓库里 varlen 是很重要的一条主线。

---

## 10. 一个最小的“理解达标线”

如果你学完后能独立讲清下面几件事，就说明你已经真正入门这个仓库了：

1. `cuLA` 为什么不是单纯的 Triton 项目
2. `fwd_o.py` 里为什么要把 CTA 划成 load / CUDA / MMA / store 几类 warp
3. `chunk_delta_h.py` 里为什么要让 `h state` 常驻寄存器
4. persistent kernel 和普通 grid launch 的核心区别是什么
5. 为什么这里很多优化都围绕 SMEM / TMEM / barrier / stage 数展开

如果还能继续回答：

6. 为什么某些 stage 只配 1，而不是一律双缓冲
7. 为什么 Blackwell 路径会显式规划 TMEM offset
8. 为什么 varlen 场景特别需要工作调度设计

那你对这个仓库的 CUDA 主线就已经很扎实了。

---

## 11. 推荐的文件阅读顺序清单

按顺序读：

1. `README.md`
2. `REPO_LAYOUT.md`
3. `setup.py`
4. `cula/lightning/la_decode.py`
5. `docs/lightning_attn_pipeline.md`
6. `docs/fwd_o_pipeline.md`
7. `cula/ops/fwd_o.py`
8. `tests/test_fwd_o.py`
9. `benchmarks/bench_fwd_o.py`
10. `docs/chunk_delta_h_pipeline.md`
11. `cula/ops/chunk_delta_h.py`
12. `tests/test_chunk_delta_h.py`
13. `benchmarks/bench_chunk_delta_h.py`
14. `cula/kda/chunk_intra.py`
15. `csrc/api/pybind.cu`
16. `csrc/kda/sm90/`
17. `csrc/kda/sm100/`

---

## 12. 最后给你的学习建议

这个仓库不太适合“从头到尾顺读一遍”。更适合的方法是：

1. 先抓一个 kernel
2. 把它的 CTA / warp / memory flow 画出来
3. 再看 benchmark 和 test
4. 最后回头看更大、更底层的 fused 实现

你会发现，真正把这个仓库读懂的关键，不是把每个 API 背下来，而是逐渐建立这种判断能力：

**当你看到一个注意力 kernel 时，你会自然去想它的 tile、并行映射、数据驻留位置、同步依赖和吞吐瓶颈。**

一旦这个视角建立起来，这个仓库就会从“很多复杂文件”变成“一组围绕 CUDA 数据流设计展开的案例集”。
