# KDA 推理后端对比: FLA、FlashKDA、cuLA、FlashInfer

本文面向一个实际选型问题: **先支持 KDA 推理，后期希望支持 LoRA 训练**。比较对象是四个本地实现:

- FLA 原生 Triton KDA: `/home/tpy/reps/flash-linear-attention`
- FlashKDA: `/home/tpy/reps/FlashKDA`
- cuLA: 当前仓库 `/home/tpy/reps/cuLA`
- FlashInfer: `/home/tpy/reps/flashinfer`

需要先区分一个容易混淆的点: 新版 FLA 的 `chunk_kda` 有 backend dispatch。如果满足 FlashKDA 的限制条件，FLA 会直接调用 `flash_kda.fwd`。因此下面把 **FLA 原生 Triton 路径** 和 **FLA + FlashKDA backend** 分开讨论。

还需要区分第二个点: **prefill 和 decode 不是同一个问题**。FlashKDA、FLA 原生 `chunk_kda`、cuLA fused/chunk KDA 主要比较的是 chunk prefill；FlashInfer 当前仓库里的 KDA 重点是 recurrent decode/spec decode，它不是 FlashKDA 那种 KDA chunk prefill 替代品。

## 一句话结论

如果当前目标是 H20/Hopper 上的 KDA 推理 prefill，且模型形状是标准 Kimi Linear 类配置，即 $K=128$、$V=128$、bf16、$HV=H$、`safe_gate=True`、`lower_bound=-5`，优先级建议是:

| 场景 | 推荐 |
|---|---|
| 纯推理 prefill，追求 H20 速度 | FlashKDA，或让 FLA dispatch 到 FlashKDA backend |
| 推理 decode，H20/Hopper | cuLA decode 或 FLA fused recurrent；FlashInfer 当前 KDA decode 要求 SM100，不适合 H20 |
| 推理 decode，Blackwell/SM100 | FlashInfer recurrent KDA 是重要候选，需和 cuLA decode 同机实测 |
| 需要通用 KDA API、训练/backward、GVA、CP、更多形状 | FLA 原生 Triton |
| 希望在 Hopper/Blackwell 上做高性能 fused kernel，并逐步补齐训练/推理能力 | cuLA |
| 后期 LoRA 训练 | 以 FLA/cuLA 的 autograd 路径为主；FlashKDA/FlashInfer 当前更偏 inference kernel |

H20/Hopper prefill forward 的理论排序:

$$
\text{FlashKDA} \approx \text{FLA + FlashKDA backend}
>
\text{cuLA fused}
>
\text{FLA 原生 Triton}
$$

decode 方向需要单独排序:

| GPU | decode 推荐判断 |
|---|---|
| H20/Hopper | 优先比较 cuLA decode 和 FLA fused recurrent；FlashInfer KDA recurrent 当前标注需要 SM100 |
| Blackwell/SM100 | FlashInfer recurrent KDA 是重量级候选；cuLA decode 也应纳入同机 benchmark |

训练/可扩展性的排序:

$$
\text{FLA 原生 Triton}
\ge
\text{cuLA}
>
\text{FlashKDA} \approx \text{FlashInfer recurrent KDA}
$$

## KDA chunk 里的核心问题

KDA 的 chunk prefill 会在 chunk 内构造下三角依赖矩阵，然后需要求类似 $I \pm L$ 的下三角逆。具体写成 $I-L$ 还是 $I+L$ 取决于代码里对严格下三角更新项 $L$ 的符号约定。直接按 token 做 forward substitution 容易有两个问题:

1. token 维度串行依赖强，指令数接近 $O(T^2)$；
2. 内存访问模式差，很难把 TensorCore/MMA 吃满。

所以 prefill 方向的三个实现都会把 chunk 内矩阵分块处理。核心块逆公式是:

$$
\begin{bmatrix}
A & 0 \\
C & D
\end{bmatrix}^{-1}
=
\begin{bmatrix}
A^{-1} & 0 \\
-D^{-1} C A^{-1} & D^{-1}
\end{bmatrix}
$$

差异主要来自四个选择:

1. chunk size 选 16、32 还是 64；
2. 逆矩阵是在 Triton 里做，还是 CUTLASS/CuTe 里做；
3. gate 的指数累计如何避免数值范围爆炸；
4. kernel 是更通用，还是为固定推理形状做强假设。

## FLA 原生 Triton KDA

源码入口是 [`fla/ops/kda/chunk.py`](../../flash-linear-attention/fla/ops/kda/chunk.py)。它是通用 KDA API，也是训练和研究最稳的基准。

### 算法结构

FLA 原生 `chunk_kda` 默认 `chunk_size=64`，也允许 32。源码里明确检查:

- 默认值: [`chunk.py`](../../flash-linear-attention/fla/ops/kda/chunk.py) 中 `chunk_size = kwargs.pop("chunk_size", 64)`
- 限制: `chunk_size` 只能是 32 或 64

forward 被拆成多段:

1. gate activation 和 chunk-local cumsum；
2. intra-chunk 计算 $A_{qk}$、$A_{kk}$ 和块逆；
3. recurrent state update；
4. output projection。

对应源码在 [`chunk_fwd.py`](../../flash-linear-attention/fla/ops/kda/chunk_fwd.py)。其中 intra kernel 使用:

$$
BT = \text{chunk\_size}, \qquad BC = 16
$$

也就是说默认 $BT=64$ 时，会把一个 64-token chunk 切成 4 个 16-token subchunk，再做 diagonal solve 和 inter-subchunk merge。

### 优势

- **通用性最好**: 支持训练/backward、GVA、variable length、context parallel 等。
- **形状限制少**: key headdim 支持到 $K \le 256$，不像 FlashKDA 固定 $K=V=128$。
- **适合作为 LoRA 训练基础**: LoRA 训练需要 backward，FLA 原生路径天然支持 autograd。
- **上游生态最好**: FLA 是很多 linear attention 变体的共同接口，后续接入模型和训练框架更容易。

### 劣势

- **推理 forward 不是最快**: 多个 Triton kernel 串起来，launch、global memory 中间结果、Triton block 调度都会有额外成本。
- **64-token chunk 的数值处理更复杂**: $lower\_bound=-5$ 时，64 步累计最坏可到 $-320$。朴素写法会遇到指数范围问题，所以 FLA 要靠 16 subchunk、fp32 diagonal buffer、局部差值和 `exp2` 等策略控制范围。
- **为了通用性牺牲了一部分专用优化空间**。

## FlashKDA

源码入口是 [`flash_kda.fwd`](../../FlashKDA/csrc/flash_kda.cpp)，文档入口是 [`README.md`](../../FlashKDA/README.md) 和 [`FlashKDA v1 deep dive`](../../FlashKDA/docs/20260420-flashkda-v1-deep-dive.md)。

### 算法结构

FlashKDA 选择:

$$
\text{CHUNK} = 16
$$

这和 FLA 默认 64 是根本差异。它把 forward 拆成两个 CUTLASS/CUDA kernel:

1. K1 token-parallel: gate activation、L2 norm、decay、构造 $L$ / $M_{qk}$、用 Neumann series 计算 16x16 三角逆；
2. K2 head-parallel: chunk recurrence、output projection、state accumulation。

因为 chunk 只有 16，严格下三角矩阵 $L$ 是幂零矩阵，满足 $L^{16}=0$。因此它不需要像 cuLA 那样做 $8 \rightarrow 16 \rightarrow 32 \rightarrow 64$ 的 block-wise inverse，而是用有限 Neumann series。按 FlashKDA 源码里的 $L$ 符号，目标等价于:

$$
(I+L)^{-1}
=
I - L + L^2 - L^3 + \cdots - L^{15}
$$

FlashKDA 代码里的写法是先构造:

$$
R_0 = I - L
$$

然后做三轮 doubling:

$$
R_1 = R_0 + R_0 L^2
$$

$$
R_2 = R_1 + R_1 L^4
$$

$$
R_3 = R_2 + R_2 L^8
$$

展开后正好得到到 $L^{15}$ 为止的有限级数。也就是说 FlashKDA 的“求逆”更准确地说是 **利用 16x16 严格三角矩阵的幂零性质，把逆写成幂矩阵有限和**，而不是通用矩阵求逆。

如果把同一问题写成 $(I-L')^{-1}$，只需要令 $L'=-L$，符号会变成全正的有限和；算法本质仍然是幂矩阵有限相加。

### 优势

- **H20 推理 prefill 速度最好**: FlashKDA 自带 H20 benchmark 中，$T=8192,H=64,D=128$ 时 fixed case 是 1.6217 ms，而 FLA 原生 `chunk_kda` 是 3.1659 ms，约 1.95x。
- **数值范围天然更稳**: 当 `lower_bound=-5`，16-token chunk 的最坏累计范围是 $-80$，比 64-token 的 $-320$ 安全得多。
- **实现路径短**: 专注 bf16、$K=V=128$、推理 forward，kernel 设计可以很激进。
- **可以作为 FLA backend 自动使用**: FLA backend verifier 通过时，`chunk_kda` 会调用 `flash_kda.fwd`。

### 劣势

- **限制非常强**:
  - inference only；
  - bf16；
  - $K=128$、$V=128$；
  - 不支持 GVA，即要求 $HV=H$；
  - 需要 `use_gate_in_kernel=True`、`use_qk_l2norm_in_kernel=True`、`use_beta_sigmoid_in_kernel=True`；
  - 需要 `safe_gate=True`；
  - 需要 `state_v_first=True`。
- **不适合作为 LoRA 训练主路径**: 当前 FlashKDA 是 forward/inference kernel，没有 backward。LoRA 训练即使只训练低秩参数，也仍然需要对 KDA 输入、投影权重或 LoRA adapter 传播梯度。
- **chunk=16 也有代价**: chunk 数量是 chunk=64 的 4 倍。FlashKDA 通过强 fusion 和 CUTLASS kernel 把这个成本压下去了，但这依赖固定形状和推理假设。

## cuLA

cuLA 是当前仓库的实现。KDA chunk 入口在 [`cula/kda/chunk.py`](../cula/kda/chunk.py)，Hopper fused forward 入口在 [`cula/kda/hopper_fused_fwd.py`](../cula/kda/hopper_fused_fwd.py)。

### 算法结构

cuLA 的 KDA chunk 和 Hopper fused forward 都使用:

$$
\text{chunk\_size} = 64
$$

它不是普通 forward substitution，而是在 CUDA/CuTe/CUTLASS 里做 64x64 的 block-wise inverse。逆矩阵工具在 [`collective_inverse.hpp`](../csrc/kerutils/include/kerutils/device/sm80/collective_inverse.hpp)，内部从小块逆逐级合并到 64x64。

对于 half 路径，大致是:

$$
8 \times 8
\rightarrow
16 \times 16
\rightarrow
32 \times 32
\rightarrow
64 \times 64
$$

对于 TF32 路径，大致是:

$$
16 \times 16
\rightarrow
32 \times 32
\rightarrow
64 \times 64
$$

### 优势

- **比 FLA 原生 Triton 更接近硬件**: 用 CuTe/CUTLASS 组织 MMA、shared memory、pipeline，适合继续做 Hopper/Blackwell 专用优化。
- **已有 H200 benchmark 优势**: 当前仓库的 H200 报告里，KDA fused forward 对 FLA Triton 平均有 1.58x speedup；固定长度大 batch、varlen 场景收益明显。
- **比 FlashKDA 更有通用潜力**: cuLA 不是只为 $K=V=128,HV=H$ 的推理 forward 服务，仓库里已有 KDA chunk forward/backward、decode、Hopper/Blackwell 多条路径。
- **适合作为后续工程主线**: 如果目标是最终同时覆盖推理和 LoRA 训练，cuLA 的方向比 FlashKDA 更接近“高性能且可训练”的内核库。

### 劣势

- **H20 推理 prefill 不一定赢 FlashKDA**: cuLA 的 chunk=64 和 block-wise 64x64 inverse 比 FlashKDA 的 16x16 inverse 更重。H200 benchmark 显示 cuLA 明显快于 FLA，但没有达到 FlashKDA 在 H20 报告里相对 FLA 的 1.9x 到 2.3x 幅度。
- **数值策略更复杂**: chunk=64 必须认真处理 gate 累计和指数缩放。cuLA Hopper fused path 要求 `safe_gate=True`，并检查 `lower_bound` 在 $[-5,0)$。
- **工程成熟度仍在早期**: README 也明确说 cuLA 还处于 early stage，API 和优化空间都还在变化；另外 Hopper/Blackwell fused prefill 当前是 forward-only，backward 仍未实现，训练要走已有 autograd/chunk backward 路径。

## FlashInfer

FlashInfer 是另一个很重要的仓库，但它当前和 FlashKDA/cuLA/FLA chunk KDA 的重心不同。KDA 相关入口是 [`flashinfer/kda_decode.py`](../../flashinfer/flashinfer/kda_decode.py)，核心实现是 [`flashinfer/kda_kernels/recurrent_kda.py`](../../flashinfer/flashinfer/kda_kernels/recurrent_kda.py)。

### 算法结构

FlashInfer 的 KDA 实现是 recurrent decode kernel。它处理的是 decode 阶段的一步或少量 speculative tokens，而不是 chunk prefill 里的 $64 \times 64$ 或 $16 \times 16$ 块逆。

其 recurrent 公式可以写成:

$$
S_t
=
\exp(g_t) \odot S_{t-1}
+
\beta_t \, k_t^\top \left(v_t - k_t S_{t-1}\right)
$$

$$
o_t = q_t S_t
$$

源码注释里也明确说 logical recurrent state 是 $S[K,V]$，kernel 中以转置布局 $[V,K]$ 的 bf16 state 存储。它支持:

- single-token decode，$T=1$；
- fused speculative decode，$T=1+\text{num\_spec\_tokens}$；
- GQA，即 $HV$ 可以是 $H$ 的倍数；
- `cu_seqlens` packed tokens；
- `ssm_state_indices` paged state/cache indices；
- pre-computed gate、softplus gate、`lower_bound * sigmoid` gate 三类 gate mode。

### 关键限制

FlashInfer `recurrent_kda` 当前限制很明确:

- 需要 SM100 / Blackwell；
- q/k/v/g/beta 都要求 bf16；
- 要求 $K=V$；
- head dim 只支持 $K \in \{64,128\}$；
- state 是 bf16，形状为 $[N,HV,V,K]$，并且会 in-place update；
- 当使用 `cu_seqlens` 时，batch 维度要求是 1。

这意味着它对 **H20/Hopper KDA decode** 不是直接候选，因为 H20 是 Hopper 系列，不满足 SM100。它更像是 Blackwell serving 场景下的 KDA decode 专用后端。

### 优势

- **decode/spec decode 很强相关**: 它不是泛泛的 linear attention 库，而是直接实现 KDA recurrent decode。
- **serving 友好**: 支持 state indices、packed tokens、speculative decode，这些都是在线推理系统真正会用到的能力。
- **CuTe DSL + JIT 开发模式**: FlashInfer 默认 JIT，改 kernel 后首次调用自动重编译，适合快速迭代。
- **和 cuLA 有技术关联**: cuLA 的 `collective_inverse.hpp` 注释里说明 half 版 collective inverse 适配自 FlashInfer 的 flat collective inverse；FlashInfer 还有 DeltaRule/GDN prefill 的 flat kernel 体系。

### 劣势

- **不是 H20 prefill 方案**: 它当前 KDA 主线是 recurrent decode，并且要求 SM100；不能拿它直接替代 FlashKDA 的 H20 chunk prefill。
- **不是训练路径**: 当前 KDA decode API 是 inference kernel，不提供 LoRA 训练所需的 autograd/backward 主路径。
- **和 FlashKDA/cuLA 的 benchmark 维度不同**: FlashKDA 比的是 prefill；FlashInfer KDA 更应该和 cuLA/FLA decode benchmark 放在一起比较。

### 对选型的影响

FlashInfer 应该被纳入设计，但位置是 **decode backend**，不是 **prefill backend**。

| 阶段 | FlashInfer 的角色 |
|---|---|
| H20 prefill | 不作为主候选，仍看 FlashKDA/FLA/cuLA |
| H20 decode | 当前 KDA recurrent 要求 SM100，不能直接用 |
| Blackwell decode | 重要候选，尤其是 speculative decode 和 paged state 场景 |
| LoRA 训练 | 不作为训练主路径，只能作为推理服务侧 kernel |

## 数值溢出问题

KDA gate 在 log space 中累计。简化看，如果每步 gate 下界是 $-5$，长度为 $C$ 的 chunk 内累计范围最坏是:

$$
\Delta g_{\max} \approx 5C
$$

所以:

$$
C=16 \Rightarrow \exp(80)
$$

$$
C=64 \Rightarrow \exp(320)
$$

如果直接在 bf16/fp32 中朴素计算 $\exp(320)$，这显然会出问题。三者的处理方式不同:

| 实现 | 是否靠小 chunk 避免 | 是否需要额外 rescale/局部差值 | 风险判断 |
|---|---:|---:|---|
| FlashKDA | 是，chunk=16 | 少 | 最自然，数值范围最舒服 |
| FLA 原生 | 否，默认 chunk=64 | 需要 | 实现已做处理，但逻辑复杂 |
| cuLA | 否，chunk=64 | 需要 | fused path 通过 safe_gate 和参考点缩放控制 |
| FlashInfer recurrent decode | 不走 chunk 逆 | decode 单步/少步 recurrent | 风险来自 gate activation 和 state 更新，不是 chunk 内指数累计 |

因此不能简单说 FLA/cuLA “会溢出”。更准确的说法是: **朴素 chunk=64 会溢出；实际 FLA/cuLA 不是朴素实现，它们通过 safe gate、subchunk、fp32 buffer、局部指数差值和 rescale 避免直接进入危险范围。FlashKDA 则通过 chunk=16 从源头降低了数值压力。**

## 对 LoRA 训练的影响

LoRA 训练不是只跑 forward。即使 base model frozen，LoRA adapter 的梯度也要从 loss 反传到 attention 输入/输出相关投影，所以 KDA kernel 至少要满足:

1. forward 输出正确；
2. backward 能给出对 $q,k,v,g,\beta$ 或相关上游张量的梯度；
3. 支持训练常见的 batch、varlen、checkpoint/recompute 策略；
4. 最好支持 fp32 initial/final state 或训练中稳定的 state 表示。

按这个标准:

| 实现 | LoRA 训练适配性 | 原因 |
|---|---|---|
| FLA 原生 Triton | 强 | 已有 autograd/backward，通用性好 |
| cuLA | 中到强 | chunk/autograd 路径已有 backward；fused prefill 当前 forward-only，训练需要按目标模型补齐测试和接口 |
| FlashKDA | 弱 | 当前定位是 inference forward backend |
| FlashInfer recurrent KDA | 弱 | 当前是 inference decode backend，不是训练/backward 路径 |

实际路线建议:

1. **短期推理**: 使用 FlashKDA 或 FLA 自动 dispatch 到 FlashKDA。这样最快，接入成本低。
2. **短期 fallback**: 当形状不满足 FlashKDA 限制时，fallback 到 FLA 原生 `chunk_kda`，保证能跑。
3. **decode 路线**: H20/Hopper 上先用 cuLA/FLA decode；Blackwell 上把 FlashInfer recurrent KDA 加入同机 benchmark。
4. **中期性能主线**: 在 cuLA 上补目标模型需要的 shape、varlen、decode、state layout 和 benchmark。
5. **LoRA 训练**: 先用 FLA 原生路径打通训练正确性；随后把热点 forward/backward kernel 替换成 cuLA。FlashKDA 和 FlashInfer 除非补 backward，否则不应作为训练主路径。

## 推荐决策

如果项目目标是“先推理，后 LoRA 训练”，推荐分层:

| 层级 | 后端 | 用途 |
|---|---|---|
| fast prefill path | FlashKDA backend | 标准 Kimi Linear/KDA 推理 prefill |
| decode path | cuLA / FLA / FlashInfer | H20 先比较 cuLA/FLA；Blackwell 加入 FlashInfer recurrent KDA |
| general fallback path | FLA 原生 Triton | 非标准 shape、训练、调试、正确性基准 |
| optimization path | cuLA | Hopper/Blackwell 高性能 forward、decode、未来 backward |

最终工程上可以做成 dispatch:

1. prefill: 若处于 inference mode，且 $K=V=128$、$HV=H$、bf16、`safe_gate=True`，优先 FlashKDA；
2. decode: 若是 Blackwell/SM100，且满足 FlashInfer `recurrent_kda` 限制，把 FlashInfer 作为候选；
3. decode: 若是 H20/Hopper，优先比较 cuLA decode 和 FLA fused recurrent；
4. 训练或 backward: 优先 FLA/cuLA autograd 路径；
5. 若 GPU 是 Hopper/Blackwell 且 cuLA 已覆盖该 shape，推理可用 cuLA 替换 FLA 原生；训练只使用已覆盖 backward 的 cuLA 路径；
6. 否则回退到 FLA 原生 Triton，保证 correctness。

## 参考文件

- FLA API 和限制: [`chunk.py`](../../flash-linear-attention/fla/ops/kda/chunk.py)
- FLA forward 拆分: [`chunk_fwd.py`](../../flash-linear-attention/fla/ops/kda/chunk_fwd.py)
- FLA intra block solve: [`chunk_intra.py`](../../flash-linear-attention/fla/ops/kda/chunk_intra.py)
- FLA FlashKDA backend: [`flashkda.py`](../../flash-linear-attention/fla/ops/kda/backends/flashkda.py)
- FlashKDA API: [`flash_kda.cpp`](../../FlashKDA/csrc/flash_kda.cpp)
- FlashKDA design note: [`20260420-flashkda-v1-deep-dive.md`](../../FlashKDA/docs/20260420-flashkda-v1-deep-dive.md)
- FlashKDA H20 benchmark: [`BENCHMARK_H20.md`](../../FlashKDA/BENCHMARK_H20.md)
- cuLA KDA chunk: [`cula/kda/chunk.py`](../cula/kda/chunk.py)
- cuLA Hopper fused forward: [`cula/kda/hopper_fused_fwd.py`](../cula/kda/hopper_fused_fwd.py)
- cuLA block inverse: [`collective_inverse.hpp`](../csrc/kerutils/include/kerutils/device/sm80/collective_inverse.hpp)
- cuLA H200 benchmark: [`BENCHMARK_H200.md`](../BENCHMARK_H200.md)
- cuLA H20 decode benchmark: [`BENCHMARK_KDA_DECODE_H203E.md`](../BENCHMARK_KDA_DECODE_H203E.md)
- FlashInfer KDA decode API: [`kda_decode.py`](../../flashinfer/flashinfer/kda_decode.py)
- FlashInfer recurrent KDA kernel: [`recurrent_kda.py`](../../flashinfer/flashinfer/kda_kernels/recurrent_kda.py)
- FlashInfer recurrent KDA benchmark: [`bench_recurrent_kda.py`](../../flashinfer/benchmarks/bench_recurrent_kda.py)
- FlashInfer flat collective inverse: [`flat_collective_inverse.hpp`](../../flashinfer/include/flashinfer/flat/ampere/collective/flat_collective_inverse.hpp)
