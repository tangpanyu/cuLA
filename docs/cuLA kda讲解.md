# cuLA KDA 讲解

本文沿用 `docs/FlashKDA Kernel 讲解.md` 的组织方式，但讲的是当前 cuLA 仓库里的 KDA 实现。这里要先纠正一个定位：cuLA 这条 `chunk_kda` 主线更像是 **FLA Triton KDA 的 CUDA/CuTe 化版本**，不是 FlashKDA 那种固定 fused prefill pipeline。

也就是说，读 cuLA 的默认 KDA 时，最合适的参照物是 `/home/tpy/reps/flash-linear-attention/fla/ops/kda/`：

- FLA 用 Triton 实现 `chunk_intra.py`、`wy_fast.py`、`chunk_fwd.py`、`chunk_bwd.py`。
- cuLA 保留同一套 chunkwise 公式和中间量名字：`Aqk`、`Akk`、`w`、`u`、`kg`、`v_new`、`h`。
- cuLA 把其中性能敏感的 forward intra / W-U recompute / state / output / backward WY 部分替换成 SM100 CUDA 或 CuTe DSL；backward intra 仍然基本是 Triton kernel。

本文重点放在默认入口：

```python
from cula.kda import chunk_kda
```

先记住一句话：

```text
cuLA 的默认 KDA 不是从 FlashKDA K1/K2 重新设计出来的单个 fused kernel；
它是对 FLA Triton chunk_kda 公式语义的替换实现：
forward 仍然拆成 gate -> intra -> W/U recompute -> state recurrence -> output；
backward 仍然反向走 output、state、W/U、intra 和 gate 的梯度链。
```

当前默认训练/推理 prefill 路径固定：

```text
chunk_size = 64
K = V = 128
q/k/v dtype = bf16
initial_state dtype = fp32
Blackwell modular forward 需要 safe_gate=True
```

代码入口：

| 文件 | 角色 |
|---|---|
| `cula/kda/__init__.py` | 导出 `chunk_kda`、SM90/SM100 prefill、decode API |
| `cula/kda/chunk.py` | FLA-compatible `chunk_kda` 入口和 autograd wrapper |
| `cula/kda/chunk_fwd.py` | 对齐 FLA `chunk_fwd.py` 的 forward 编排 |
| `cula/kda/chunk_bwd.py` | 对齐 FLA `chunk_bwd.py` 的 backward 编排 |
| `cula/kda/chunk_intra.py` | forward intra CUDA wrapper 和 backward intra Triton kernel |
| `csrc/api/kda_sm100.cu` | SM100 CUDA 扩展参数打包 |
| `csrc/kda/sm100/` | SM100 C++/CUTLASS KDA forward intra 和 `recompute_w_u` |
| `cula/ops/chunk_delta_h_sm100.py` | CuTe DSL chunk state recurrence forward |
| `cula/ops/fwd_o_sm100.py` | CuTe DSL output kernel |
| `cula/ops/chunk_wy_dqkg_sm100.py` | CuTe DSL backward fused W/Y/dq/dk/dg/db kernel |
| `cula/ops/kda_decode.py` | 单 token decode kernel |

和 FLA 的直接对应关系是：

| FLA Triton 路径 | cuLA 对应路径 | 语义 |
|---|---|---|
| `fla/ops/kda/chunk_fwd.py` | `cula/kda/chunk_fwd.py` | forward 编排，公式分块相同 |
| `fla/ops/kda/chunk_intra.py` | `cula/kda/chunk_intra.py` + `csrc/kda/sm100/` | 生成 `Aqk/Akk` |
| `fla/ops/kda/wy_fast.py::recompute_w_u_fwd` | `cula_cuda.recompute_w_u_cuda` | 从 `Akk` 得到 `w/u/kg/qg` |
| `fla/ops/common/chunk_delta_h.py` | `cula/ops/chunk_delta_h_sm100.py` | chunk 间 state recurrence |
| `fla/ops/gla/chunk.py::chunk_gla_fwd_o_gk` | `cula/ops/fwd_o_sm100.py` | 合成 output |
| `fla/ops/kda/chunk_bwd.py` | `cula/kda/chunk_bwd.py` | backward 编排 |
| `fla/ops/kda/chunk_intra.py::chunk_kda_bwd_kernel_intra` | `cula/kda/chunk_intra.py::chunk_kda_bwd_kernel_intra` | backward intra，Triton 逻辑基本保留 |

## 1. 从 KDA 单步到 chunk 公式

KDA 的 recurrent state 是固定大小矩阵：

$$
S_t\in\mathbb{R}^{d_k\times d_v}
$$

单步更新可以写成：

$$
\bar S_t=\operatorname{Diag}(\exp(g_t))S_{t-1}
$$

$$
D_t=\beta_t(v_t-\bar S_t^\top k_t)
$$

$$
S_t=\bar S_t+k_tD_t^\top
$$

$$
o_t=S_t^\top q_t
$$

KDA 和 GDN 的关键差异是 decay。GDN 的 decay 是 scalar；KDA 的 $g_t$ 是 key/channel-wise vector，所以 chunk 内 score 不能写成普通 $QK^\top$ 外乘一个 scalar mask，而要把相对 decay 放到 dot product 里面。

对一个 chunk，令 $BT=64$：

$$
Q,K,G\in\mathbb{R}^{BT\times d_k},\qquad
V,O,D\in\mathbb{R}^{BT\times d_v}
$$

进入 chunk 后，raw gate 做 chunk-local cumsum：

$$
g_r=\sum_{j=0}^{r}g^{raw}_j
$$

当前代码会把它转换到 base-2 log 域：

$$
g_r^{(2)}=\log_2(e)\,g_r
$$

因此 kernel 里用：

$$
\Gamma_r=2^{g_r^{(2)}}
$$

然后定义：

$$
Q_\gamma=\Gamma\odot Q
$$

$$
K_\gamma=\Gamma\odot K
$$

$$
K_{\gamma^{-1}}=K/\Gamma
$$

$$
K_{\text{right}}=(\Gamma_{BT-1}/\Gamma)\odot K
$$

chunk 内两个核心矩阵如下。注意 `scale` 在 cuLA 实现里显式参与 output 路径：`fwd_o` 对旧 state 项乘 `scale`，`chunk_intra` 写出的 `Aqk` 也带 `scale`。

$$
B=K_\gamma K_{\gamma^{-1}}^\top
$$

$$
E=scale\cdot\operatorname{Tril}(Q_\gamma K_{\gamma^{-1}}^\top)
$$

三角解矩阵是：

$$
A=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
\operatorname{Diag}(\beta)
$$

有了 $A$ 后：

$$
W=A K_\gamma
$$

$$
U=A V
$$

$$
D=U-WS_0=A(V-K_\gamma S_0)
$$

输出和 state 更新为：

$$
O=scale\cdot Q_\gamma S_0+ED
$$

$$
S_C=\operatorname{Diag}(\Gamma_{BT-1})S_0+K_{\text{right}}^\top D
$$

这就是 cuLA modular forward/backward 都围绕的数学骨架。

## 2. cuLA 为什么沿用 FLA 的模块拆分

FLA 的 Triton `chunk_kda` 没有把所有事情塞进一个 kernel，而是按 chunkwise KDA 公式拆成多个阶段。cuLA 的默认 `chunk_kda` 沿用了这套拆分：

```text
gate cumsum
-> chunk intra: 生成 Aqk/Akk
-> recompute_w_u: 生成 W/U/K_right
-> chunk_delta_h: 顺序推进 state，生成 D/v_new 和每个 chunk 的 h
-> fwd_o: 合成 output
```

原因是这些阶段的并行形态不同：

- `chunk_intra` 是 chunk 内矩阵构造和下三角 inverse，按 chunk/head 并行；FLA 用 Triton，cuLA forward 用 SM100 CUDA。
- `recompute_w_u` 是 $A@K_\gamma$ 和 $A@V$ 的矩阵乘；FLA 用 Triton `wy_fast.py`，cuLA forward 用 `recompute_w_u_cuda`。
- `chunk_delta_h` 沿 sequence 的 chunk 顺序推进同一个 state，有时间依赖；cuLA 用 CuTe DSL 版本替换 FLA common kernel。
- `fwd_o` 再次是 chunk/head 并行，可以单独优化 output 合成；cuLA 用 CuTe DSL output kernel。

所以本文讲的不是 FlashKDA K1/K2 的 fused pipeline，而是 FLA Triton KDA 的同构实现：公式和中间量沿用 FLA，具体 kernel 后端在 cuLA 里被替换。

## 3. Python 入口：`chunk_kda`

用户调用：

```python
o, final_state = chunk_kda(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5,
    A_log=A_log,
    dt_bias=dt_bias,
)
```

入口形状：

$$
q,k\in\mathbb{R}^{B\times T\times H\times K}
$$

$$
v,o\in\mathbb{R}^{B\times T\times HV\times V}
$$

$$
g\in\mathbb{R}^{B\times T\times HV\times K}
$$

$$
\beta\in\mathbb{R}^{B\times T\times HV}
$$

$$
S_0,S_T\in\mathbb{R}^{N\times HV\times K\times V}
$$

GVA 下，$HV$ 可以大于 $H$。代码用：

```text
G = HV // H
qk_head = value_head // G
```

也就是说 `q/k` 只有 $H$ 个 head，而 `v/g/beta/Aqk/Akk/w/u/kg` 都在 $HV$ 个 value-head 空间里。

`ChunkKDAFunction.forward` 保存 backward 需要的中间量：

```text
q, k, v, g, g_org, beta, A_log, dt_bias,
Aqk, Akk, w, u, qg, kg, v_new, h, initial_state,
cu_seqlens, chunk_indices
```

如果 `disable_recompute=False`，forward 结束前会释放 `w/u/qg/kg/v_new`，backward 再重算。这样省显存。若 `disable_recompute=True`，这些中间量会保留，backward 少做一遍 recompute。

## 4. gate 和 chunk index

forward 先处理 gate：

```python
if use_gate_in_kernel:
    g = kda_gate_chunk_cumsum(
        g=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
else:
    g = chunk_local_cumsum(
        g=g,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
```

`g` 后续都是 chunk-local cumulative log decay，并且已经乘过 $1/\ln 2$。所以代码里统一用 `exp2`：

$$
\exp(g_{\text{natural}})=2^{g_{\text{natural}}/\ln 2}
$$

变长输入通过 `cu_seqlens` 和 `chunk_indices` 表示。`prepare_chunk_indices(cu_seqlens, 64)` 会生成每个 chunk 对应的 `(sequence_id, local_chunk_id)`，后续 forward/backward kernel 都用这张表定位 token range。

## 5. Forward 按公式分块总览

cuLA 的 forward 可以按公式拆成五块。先不要看函数名，先看每一块在数学上留下什么中间量：

```text
F0: gate cumsum
    g_raw -> g -> Gamma

F1: intra score 和三角 inverse
    Q_gamma, K_gamma, K_inv -> B, E, M

F2: 把三角解作用到 K/V
    M, beta, K_gamma, V -> W, U, K_right

F3: 沿 chunk 更新 state
    W, U, K_right, S0 -> D, S_C

F4: 合成输出
    Q_gamma, S0, E, D -> O
```

对应到 FLA Triton 语义和 cuLA 替换点：

| 公式块 | 数学中间量 | FLA Triton 路径 | cuLA 路径 |
|---|---|---|---|
| F0 | $g,\Gamma$ | `kda_gate_chunk_cumsum` / `chunk_local_cumsum` | 直接复用 FLA gate/cumsum |
| F1 | $E,M$ | `fla/ops/kda/chunk_intra.py` | `cula_cuda.chunk_kda_fwd_intra_cuda` |
| F2 | $W,U,K_{\text{right}}$ | `fla/ops/kda/wy_fast.py::recompute_w_u_fwd` | `cula_cuda.recompute_w_u_cuda` |
| F3 | $D,S_C$ | `fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_h` | `cula/ops/chunk_delta_h_sm100.py` |
| F4 | $O$ | `fla.ops.gla.chunk.chunk_gla_fwd_o_gk` | `cula/ops/fwd_o_sm100.py` |

这里 `Akk` 不是完整的 $A$。当前实现把完整三角解拆成：

$$
M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
$$

$$
A=M\operatorname{Diag}(\beta)
$$

`Akk` 保存 $M$，`recompute_w_u` 再把 $\operatorname{Diag}(\beta)$ 乘到 $K_\gamma$ 和 $V$ 上。

## 6. Forward F0：gate 变成 chunk 内累计 decay

输入的 `g` 是 raw log-space gate，或者是已经预处理好的 log decay。forward 第一块把它变成 chunk-local cumulative gate：

$$
g_r=\sum_{j=0}^{r}g^{raw}_j
$$

cuLA 为了使用 `exp2`，把自然指数 log 域转到 base-2：

$$
\tilde g_r=\log_2(e)\,g_r
$$

后面 kernel 里真正用的是：

$$
\Gamma_r=2^{\tilde g_r}
$$

这一块输出的 `g` 已经不是 raw gate，而是 $\tilde g$。因此后面看到 `exp2(g)` 时，数学语义是 $\exp(g_{\text{natural}})$。

代码对应：

| 情况 | 代码 | 输出语义 |
|---|---|---|
| kernel 内做 gate activation | `kda_gate_chunk_cumsum` | $\tilde g=\log_2(e)\cdot\operatorname{cumsum}(gate(g_{raw}))$ |
| 外部已给 log decay | `chunk_local_cumsum` | $\tilde g=\log_2(e)\cdot\operatorname{cumsum}(g_{raw})$ |

## 7. Forward F1：构造 $B$、`Aqk` 和 `Akk`

这一块只看一个 chunk，形状是：

$$
Q,K\in\mathbb{R}^{BT\times K},\qquad BT=64
$$

先构造三个带 decay 的矩阵：

$$
Q_\gamma=\Gamma\odot Q
$$

$$
K_\gamma=\Gamma\odot K
$$

$$
K_{\gamma^{-1}}=K/\Gamma
$$

它们的作用是把 channel-wise 相对 decay 放进 dot product：

$$
(Q_\gamma K_{\gamma^{-1}}^\top)_{r,i}
=
\sum_c q_{r,c}\,\Gamma_{r,c}\,k_{i,c}/\Gamma_{i,c}
=
\sum_c q_{r,c}\,k_{i,c}\,\exp(g_{r,c}-g_{i,c})
$$

然后算两个 chunk 内 score：

$$
B=K_\gamma K_{\gamma^{-1}}^\top\in\mathbb{R}^{BT\times BT}
$$

$$
E_{\text{raw}}=\operatorname{Tril}(Q_\gamma K_{\gamma^{-1}}^\top)\in\mathbb{R}^{BT\times BT}
$$

`Aqk` 保存的是带 `scale` 的 $E$：

$$
Aqk=E=scale\cdot E_{\text{raw}}
$$

`Akk` 保存的是下三角 solve 的 inverse core：

$$
Akk=M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
$$

这里的 strict lower triangular 来自 causal delta-rule 依赖：第 $r$ 个 token 的有效写入依赖前面 $i<r$ 的写入，不依赖自己和未来 token。

这块和 FLA 的 Triton 实现是一一对应的。FLA 的 `safe_gate=True` 路径先用 `chunk_kda_fwd_kernel_intra_sub_chunk` 处理每个 16-token diagonal sub-chunk，再用 `chunk_kda_fwd_kernel_inter_solve_fused` 补齐 off-diagonal sub-chunk 并拼出完整 $BT\times BT$ 的 inverse。

cuLA forward 仍然用了 16-token subchunk 思路，但它不是 FLA 那种两个 Triton kernel 加 `Akkd` 临时 diagonal block 的外显结构。SM100 forward intra 在 `csrc/kda/sm100/kda_fwd_intra_mainloop_sm100.hpp` 里固定：

```cpp
static constexpr int SubTileT = 16;
static constexpr int TileT = 64;
```

也就是把一个 $64\times64$ chunk 看成 $4\times4$ 个 16-token subchunk block。代码注释里也直接写了 lower-triangular 4×4 subchunk matrix：

```text
intra[0]
inter[0] intra[1]
inter[1] inter[2] intra[2]
inter[3] inter[4] inter[5] intra[3]
```

所以准确说法是：cuLA forward 不再跑 FLA 的 `chunk_kda_fwd_kernel_intra_sub_chunk` / `chunk_kda_fwd_kernel_inter_solve_fused` 两个 Triton kernel，而是在一个 SM100 CUDA/CUTLASS kernel 内部用 `SubTileT=16` 的 subchunk tiling 生成同样语义的 `Aqk/Akk`。

代码做的事情可以按公式对应为：

| 公式动作 | FLA Triton 变量 | cuLA 变量 | 说明 |
|---|---|---|---|
| $scale\cdot\operatorname{Tril}(Q_\gamma K_{\gamma^{-1}}^\top)$ | `Aqk` | `Aqk` | output 的 chunk 内读取矩阵 |
| $\left[I+\operatorname{StrictTril}(\operatorname{Diag}(\beta)B)\right]^{-1}$ | `Akk` | `Akk` | 后续求 $W/U$ 的 inverse core |

FLA 代码里为了数值稳定，会在 16-token sub-chunk 内选一个 `b_gn` 作为 reference，实际算的是：

$$
(g_r-b_gn) + (b_gn-g_i)=g_r-g_i
$$

所以它不是改变公式，而是把相对 decay 拆成两边乘：

$$
q_r\exp(g_r-b_gn)\cdot k_i\exp(b_gn-g_i)
=q_rk_i\exp(g_r-g_i)
$$

cuLA 的 forward intra 也保留这个相对 decay 语义，只是硬件实现从 Triton `tl.dot` 换成 SM100 UMMA/TMEM，并由 CUDA-side inverse 逻辑处理 mask、beta 和 $64\times64$ 下三角 inverse。

## 8. Forward F2：把三角解作用到 $K_\gamma$ 和 $V$

简化公式里常把完整三角解写成：

$$
A=M\operatorname{Diag}(\beta)
$$

然后：

$$
W=A K_\gamma
$$

$$
U=A V
$$

FLA 的 Triton `wy_fast.py::recompute_w_u_fwd_kda_kernel` 和 cuLA 的 `recompute_w_u_cuda` 做的是同一件事：都没有显式 materialize $A$，而是直接构造：

$$
\operatorname{Diag}(\beta)K_\gamma
$$

$$
\operatorname{Diag}(\beta)V
$$

再左乘 `Akk=M`：

$$
w=M\left(\operatorname{Diag}(\beta)K_\gamma\right)=W
$$

$$
u=M\left(\operatorname{Diag}(\beta)V\right)=U
$$

同一个 kernel 还生成 state 更新需要的 right-decayed key。FLA 里这个变量也叫 `kg`，cuLA 继续沿用这个名字：

$$
kg=K_{\text{right}}=(\Gamma_{BT-1}/\Gamma)\odot K
$$

以及可选的：

$$
qg=Q_\gamma
$$

变量对照：

| 代码变量 | 数学含义 | 后续用途 |
|---|---|---|
| `w` | $W=A K_\gamma$ | 算 $D=U-WS_0$ |
| `u` | $U=AV$ | 算 $D=U-WS_0$ |
| `kg` | $K_{\text{right}}$ | 更新 $S_C$ |
| `qg` | $Q_\gamma$ | backward 或保存中间量 |

`kg` 的名字容易误导。它不是 $K_\gamma$，而是：

$$
K_{\text{right}}=(\Gamma_{BT-1}/\Gamma)\odot K
$$

## 9. Forward F3/F4：更新 state 并合成输出

### F3: state recurrence

每个 chunk 开始时，当前 state 记为：

$$
S_0\in\mathbb{R}^{K\times V}
$$

cuLA 会把这个 chunk-start state 保存到 `h`，供 output 和 backward 使用：

$$
h=S_0
$$

有效写入是：

$$
D=U-WS_0
$$

代码变量：

$$
v\_new=D
$$

然后更新 chunk 末尾 state：

$$
S_C=\operatorname{Diag}(\Gamma_{BT-1})S_0+K_{\text{right}}^\top D
$$

在代码里：

| 数学项 | 代码变量 |
|---|---|
| $S_0$ | `h` 中保存的 chunk-start state |
| $D$ | `v_new` |
| $K_{\text{right}}$ | `kg` |
| $S_C$ | 下一 chunk 的 state / `final_state` |

### F4: output

output 有两部分：

$$
O_{\text{state}}=scale\cdot Q_\gamma S_0
$$

$$
O_{\text{intra}}=E D
$$

所以：

$$
O=scale\cdot Q_\gamma S_0+E D
$$

其中 `Aqk` 已经保存了 $E=scale\cdot E_{\text{raw}}$，所以代码里直接做：

$$
O_{\text{intra}}=Aqk\cdot v\_new
$$

这就是 `fwd_o` reference 里的两项：

```python
qg = q_chunk * (2.0 ** g_chunk)
o_inter = scale * (qg @ h_state)
o_intra = tril(Aqk) @ v_new
o = o_inter + o_intra
```

到这里，forward 完成。完整公式链是：

$$
M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)K_\gamma K_{\gamma^{-1}}^\top\right)
\right]^{-1}
$$

$$
W=M\operatorname{Diag}(\beta)K_\gamma,\qquad
U=M\operatorname{Diag}(\beta)V
$$

$$
D=U-WS_0
$$

$$
S_C=\operatorname{Diag}(\Gamma_{BT-1})S_0+K_{\text{right}}^\top D
$$

$$
O=scale\cdot Q_\gamma S_0+
scale\cdot\operatorname{Tril}(Q_\gamma K_{\gamma^{-1}}^\top)D
$$

## 10. Backward 总览

cuLA backward 也沿用 FLA `fla/ops/kda/chunk_bwd.py` 的梯度分块方式。区别是：cuLA 把其中 `wy_dqkg_fused` 换成 CuTe DSL，把 recompute 换成 CUDA extension；但 `dAv` 和 intra backward 仍然是 Triton 风格 kernel，尤其 `cula/kda/chunk_intra.py::chunk_kda_bwd_kernel_intra` 基本就是 FLA intra backward 的同构实现。

`cula/kda/chunk_bwd.py::chunk_kda_bwd` 的主线是：

```text
如果 forward 没保存 w/u/qg/kg/v_new/h，则先 recompute
-> dAv: 从 output intra 项得到 dAqk 和一部分 dv_new
-> bwd_dhu: 反传 state recurrence，得到 dh/dh0 和更新后的 dv_new 梯度
-> bwd_wy_dqkg_fused: 反传 W/U/K_right/Q_gamma 相关路径，得到 dq/dk/dv/db/dg/dAkk
-> bwd_intra: 反传 Aqk/Akk 的 chunk 内 score 和 inverse，累加 dq/dk/db/dg
-> GVA reduce: 把 HV 维的 dq/dk 聚合回 H 维
-> reverse cumsum 和 gate bwd
```

对应到 FLA/cuLA：

| 反向公式块 | FLA 路径 | cuLA 路径 |
|---|---|---|
| $O_{\text{intra}}=AqkD$ 的反传 | `chunk_kda_bwd_dAv` Triton | `cula/kda/chunk_bwd.py::chunk_kda_bwd_dAv` Triton |
| state recurrence 反传 | `chunk_gated_delta_rule_bwd_dhu` | 直接复用 FLA common backward |
| $W/U/K_{\text{right}}/Q_\gamma$ 反传 | FLA `chunk_kda_bwd_wy_dqkg_fused` | `cula/ops/chunk_wy_dqkg_sm100.py` CuTe DSL |
| `Aqk/Akk` intra score 反传 | FLA `chunk_kda_bwd_kernel_intra` | cuLA 同名 Triton kernel |
| gate 反传 | FLA gate backward | 直接复用 FLA gate backward |

返回：

```text
dq, dk, dv, db, dg, dh0, dA, dbias
```

其中 `dA/dbias` 只在 `use_gate_in_kernel=True` 时来自 `kda_gate_bwd`。

## 11. Backward: recompute 中间量

如果 `disable_recompute=False`，forward 为了省显存已经释放了：

```text
w, u, qg, kg, v_new, h
```

backward 会重新做：

```python
g = kda_gate_chunk_cumsum(...)  # if use_gate_in_kernel
cula_cuda.recompute_w_u_cuda(...)
h, v_new, _ = chunk_gated_delta_rule_fwd_h(...)
```

也就是说 backward 开头重建 forward 的中间状态：

$$
W,\quad U,\quad K_{\text{right}},\quad Q_\gamma,\quad D,\quad h
$$

如果 `disable_recompute=True`，这些 tensor 是 forward 保存下来的，backward 直接从 `ctx.saved_tensors` 取。

## 12. Backward: `chunk_kda_bwd_dAv`

`chunk_kda_bwd_dAv` 反传 output 的 intra 项：

$$
O_{\text{intra}}=Aqk\cdot v_{\text{new}}
$$

代码注释写得很直接：

```python
# dAqk = do @ v.T
# dv = A @ do
```

这里函数实际传入的 `v` 是 `v_new`：

```python
dAqk, dv = chunk_kda_bwd_dAv(
    v=v_new,
    do=do,
    A=Aqk,
)
```

所以它输出：

| 输出 | 语义 |
|---|---|
| `dAqk` | output 对 $E/Aqk$ 的梯度 |
| `dv` | output 对 $D/v_new$ 的第一部分梯度 |

这个 kernel 是 Triton 实现，按 `(chunk, B*HV)` 并行。因为 forward 的 `Aqk` 带 `scale`，这里传给后续 intra backward 的 `dAqk` 也会乘上 `scale`，对齐到未缩放的 $Q_\gamma K_{\gamma^{-1}}^\top$ score 梯度。

## 13. Backward: `chunk_gated_delta_rule_bwd_dhu`

接着反传 state recurrence：

```python
dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
    q=qg,
    k=kg,
    w=w,
    gk=g,
    h0=initial_state,
    dht=dht,
    do=do,
    dv=dv,
    use_exp2=True,
)
```

这一步处理 forward 中：

$$
D=U-WS_0
$$

$$
S_C=\operatorname{Diag}(\Gamma_C)S_0+K_{\text{right}}^\top D
$$

以及 output inter 项：

$$
O_{\text{inter}}=Q_\gamma S_0
$$

它输出：

| 输出 | 语义 |
|---|---|
| `dh` | 每个 chunk 开始 state `h` 的梯度 |
| `dh0` | initial_state 的梯度 |
| `dv` | 累加 state recurrence 后的 `v_new/D` 梯度 |

这一步当前直接复用 FLA 的 `chunk_gated_delta_rule_bwd_dhu`，不是 cuLA 自己的 SM100 CuTe DSL kernel。

## 14. Backward: `chunk_kda_bwd_wy_dqkg_fused`

然后调用 cuLA 的 SM100 CuTe DSL kernel：

```python
dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused_cutedsl(
    q=q,
    k=k,
    v=v,
    v_new=v_new,
    g=g,
    beta=beta,
    A=Akk,
    h=h,
    do=do,
    dh=dh,
    dv=dv,
)
```

它负责反传这些 forward 关系：

$$
Q_\gamma=Q\odot 2^g
$$

$$
K_{\text{right}}=K\odot 2^{g_{BT-1}-g}
$$

$$
W=M\operatorname{Diag}(\beta)K_\gamma
$$

$$
U=M\operatorname{Diag}(\beta)V
$$

$$
D=U-WS_0
$$

CuTe DSL kernel 里用 TMEM 分区存中间 accumulator，例如：

```text
TMEM_DA_ACC_OFF      dA accumulator
TMEM_DQ_ACC_OFF      dq accumulator
TMEM_DK_ACC_OFF      dk accumulator
TMEM_DW_ACC_OFF      dw accumulator
TMEM_DKGB_ACC_OFF    A @ dw accumulator
TMEM_DQ_SCALED_OFF   保存缩放后的 dq，后面算 dg 会复用
```

几个核心反传块可以按代码注释理解：

```text
dq += do @ h
dvb = A @ dv
dw += dv @ h
dA += dv @ v^T
dk += v_new @ dh
dkgb = A @ dw
dA += dw @ kg^T
dA2 = dA @ A
dA3 = A @ dA2
```

这个 kernel 还负责：

- 对 `dq` 乘 $2^g$ 和 `scale`。
- 对 `dk` 乘 $2^{g_{BT-1}-g}$。
- 计算 `db`，并用 shared memory 固定归约顺序，避免 `atomicAdd` 带来的非确定性。
- 计算 `dg`，包括来自 $Q_\gamma$、$K_{\text{right}}$、$K_\gamma$、chunk last gate 的贡献。
- 输出 `dAkk`，交给下一步 intra backward 继续反传到 `q/k/g/beta`。

## 15. Backward: `chunk_kda_bwd_intra`

`chunk_kda_bwd_intra` 接收两路矩阵梯度：

```python
dAqk  # 来自 output intra: Aqk = scale * E_raw
dAkk  # 来自 W/U path: M = Akk
```

它反传 chunk 内两个矩阵。这里把未缩放的 query-key score 记为 $E_{\text{raw}}$，forward 写出的 `Aqk` 是 $scale\cdot E_{\text{raw}}$：

$$
E_{\text{raw}}=\operatorname{Tril}(Q_\gamma K_{\gamma^{-1}}^\top)
$$

$$
M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
$$

其中：

$$
B=K_\gamma K_{\gamma^{-1}}^\top
$$

这个 kernel 是 Triton 写的，文件在 `cula/kda/chunk_intra.py`。它按：

```text
grid = (NK * NC, NT, B * HV)
BC = min(16, BT)
BK = min(32, next_power_of_2(K))
```

把一个 64-token chunk 再切成 16-token sub-block。这样 backward intra 可以分块处理：

- `dAqk` 对 $Q_\gamma K_{\gamma^{-1}}^\top$ 的贡献。
- `dAkk` 对 $K_\gamma K_{\gamma^{-1}}^\top$ 和 lower-triangular inverse 的贡献。
- `dq/dk/dg/db` 的累加。

`safe_gate=True` 时，kernel 用更快的分块路径；否则走逐列循环处理相对 decay。

输出：

```text
dq, dk, db, dg
```

这些会和前面 `chunk_kda_bwd_wy_dqkg_fused` 的输出合并。

## 16. Backward: GVA reduce 和 gate backward

由于 backward 中 `dq/dk` 先在 value-head 空间里生成：

$$
dq,dk\in\mathbb{R}^{B\times T\times HV\times K}
$$

但原始 `q/k` 是：

$$
q,k\in\mathbb{R}^{B\times T\times H\times K}
$$

所以 GVA 下需要把每组 value heads 聚合回 qk head：

```python
if HV > H:
    dq = dq.view(B, T, H, G, K).sum(dim=3)
    dk = dk.view(B, T, H, G, K).sum(dim=3)
```

`dg` 还要从 cumulative gate 的梯度变回 raw gate 的梯度。因为 forward 做了 chunk-local cumsum，backward 做 reverse cumsum：

```python
dg = chunk_local_cumsum(
    dg,
    chunk_size=chunk_size,
    reverse=True,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
)
```

如果 `use_gate_in_kernel=True`，还要继续反传 gate activation：

```python
dg, dA, dbias = kda_gate_bwd(
    g=g_org,
    A_log=A_log,
    dt_bias=dt_bias,
    dyg=dg,
    lower_bound=lower_bound,
)
```

最终 backward 返回：

```text
dq, dk, dv, db, dg, dh0, dA, dbias
```

## 17. Decode 路径

`chunk_kda` 是 prefill/training 的 chunkwise path。decode 一次只处理一个 token，不需要 $BT\times BT$ 的 `Aqk/Akk`，直接执行单步 recurrence。

入口：

```python
from cula.kda import kda_decode
from cula.kda import fused_sigmoid_gating_delta_rule_update
```

文件：

```text
cula/ops/kda_decode.py
```

decode 逻辑仍然是：

$$
\bar S_t=\operatorname{Diag}(\exp(g_t))S_{t-1}
$$

$$
D_t=\beta_t(v_t-\bar S_t^\top k_t)
$$

$$
S_t=\bar S_t+k_tD_t^\top
$$

$$
o_t=S_t^\top q_t
$$

实现重点不再是 chunk inverse，而是：

- state layout：默认兼容 FLA 的 VK layout，也支持 KV layout。
- 小 batch / 大 batch 分流。
- 小 batch 时用 split-state variant 增加 CTA 数，缓解 GPU underfill。
- 编译缓存，避免每次调用重复编译 CuTe DSL kernel。

## 18. Fused prefill 路径

仓库里还有 fused prefill API：

```python
from cula.kda import kda_prefill_hopper
from cula.kda import kda_prefill_blackwell
```

其中：

- `cula/kda/hopper_fused_fwd.py` 是 SM90 Hopper fused prefill，走 `csrc/kda/sm90/`。
- `cula/kda/blackwell_fused_fwd.py` 调用 `cula/ops/kda_fully_fused_sm100_wip.py`，从文件名和 pyproject 标记看仍是 WIP 路线。

本文主线讲的是默认 `chunk_kda` modular path。读 fused prefill 时，可以继续沿用本文公式，但工程结构会更接近 FlashKDA 文档里的 fused pipeline。

## 19. 常见误读

### 19.1 `Akk` 不是完整的 $A$

简化版常写：

$$
A=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
\operatorname{Diag}(\beta)
$$

当前代码里 `Akk` 更接近：

$$
M=
\left[
I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)B\right)
\right]^{-1}
$$

完整 $A=M\operatorname{Diag}(\beta)$ 在 `recompute_w_u` 里通过构造 $\operatorname{Diag}(\beta)K_\gamma$ 和 $\operatorname{Diag}(\beta)V$ 完成。

### 19.2 `kg` 不是 $K_\gamma$

`kg` 在 `chunk_delta_h` 里作为 `k` 传入，但语义是：

$$
K_{\text{right}}=(\Gamma_{BT-1}/\Gamma)\odot K
$$

它用于：

$$
S_C=\operatorname{Diag}(\Gamma_{BT-1})S_0+K_{\text{right}}^\top D
$$

### 19.3 `g` 已经是 base-2 cumulative gate

进入主要 kernel 后，`g` 不是 raw gate，而是 chunk-local cumsum 后又乘了 $1/\ln 2$ 的值。实现里用 `exp2(g)`，不是 `exp(g)`。

### 19.4 forward 的 `v_new` 是有效写入 $D$

`v_new` 不是原始 value。它对应：

$$
D=U-WS_0
$$

它会被 `fwd_o` 用于 $ED$，也会被 backward 用于 `dAqk` 和 state 反传。

### 19.5 backward 混合了多套实现

当前 backward 不是单一 CuTe DSL kernel：

| 阶段 | 实现 |
|---|---|
| `dAqk/dv` | Triton |
| `dhu/dh0` | FLA common kernel |
| `dq/dk/dv/db/dg/dAkk` | cuLA SM100 CuTe DSL |
| intra `dAqk/dAkk -> dq/dk/db/dg` | Triton |
| gate backward | FLA KDA gate helper |

## 20. 读代码路线

Forward 推荐顺序：

```text
cula/kda/__init__.py
-> cula/kda/chunk.py
-> cula/kda/chunk_fwd.py
-> cula/kda/chunk_intra.py
-> csrc/api/kda_sm100.cu
-> csrc/kda/sm100/kda_fwd_intra_kernel_sm100.hpp
-> csrc/kda/sm100/kda_fwd_recomp_w_u_kernel_sm100.hpp
-> cula/ops/chunk_delta_h_sm100.py
-> cula/ops/fwd_o_sm100.py
```

Backward 推荐顺序：

```text
cula/kda/chunk.py
-> cula/kda/chunk_bwd.py
-> cula/ops/chunk_wy_dqkg_sm100.py
-> cula/kda/chunk_intra.py
-> fla.ops.common.chunk_delta_h
-> fla.ops.kda.gate
```

调试时先在 Python 编排层确认 tensor shape 和变量语义，再进入 SM100/CuTe DSL kernel 看具体的 TMA、TMEM、UMMA 和 barrier 组织。
