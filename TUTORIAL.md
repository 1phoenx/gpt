# 从零手写一个 GPT：本仓库教程

目标：读懂当前仓库，并能在不复制核心实现的情况下，自己写出一个可以训练和生成文本的 decoder-only Transformer。

建议学习方法：每读完一节，关闭文档，自己重写对应模块；只在测试失败时回来对照实现。

本教程对应 `experiment` 分支。训练入口使用 `cs336_basics/gpt.py`，课程作业接口主要位于 `cs336_basics/11.py` 和 `tests/adapters.py`。

## 1. 最终系统

```text
文本 -> 预分词 -> Byte-Level BPE -> token IDs
     -> 随机取 [input, target] 窗口
     -> token embedding
     -> Transformer blocks
          RMSNorm -> causal multi-head attention -> residual
          RMSNorm -> SwiGLU -> residual
     -> final RMSNorm -> vocabulary logits
     -> next-token cross entropy -> AdamW
     -> autoregressive sampling -> decode
```

当前实验已经实际训练过一个 `10,160,960` 参数模型。它是教学实验，不是 benchmark 配置。

## 2. 先认识目录

```text
cs336_basics/
  pretokenization.py   GPT-2 风格预分词和 BPE 训练
  tokenizer.py         BPE encode/decode
  prepare_data.py      文本编码为 .npy token 数组
  gpt.py               当前可训练 PyTorch GPT
  training.py          batch、loss、AdamW、学习率、checkpoint
  train.py             完整训练命令行入口
  pipeline.py          单步训练和文本生成
  11.py                课程作业风格的底层函数骨架

experiments/
  make_tiny_corpus.py  生成 CPU 快速实验语料
  run_experiments.sh   训练 baseline 和三个消融
  plot_results.py      生成 SVG 曲线
  write_report.py      生成消融表和技术说明

tests/
  adapters.py          作业测试期望的函数接口
  test_model.py        attention、RoPE、block、LM 测试
  test_data.py         get_batch 测试
  test_optimizer.py    AdamW 和学习率测试
  test_tokenizer.py    tokenizer 测试
  test_train_bpe.py    BPE 训练测试
```

### 三套代码的关系

1. `11.py` 的 `run_linear`、`run_rope`、`run_transformer_lm` 等函数是课程作业接口；许多函数仍是 `NotImplementedError`，测试通过 `tests/adapters.py` 调用它们。
2. `gpt.py` 是本实验真正训练的 `torch.nn.Module`。`train.py` 导入的是 `GPTLanguageModel`，不是 `11.py`。
3. `model.py` 当前分支不存在，不要把测试中的作业接口误认为训练入口。

推荐阅读顺序：

```text
gpt.py -> training.py -> train.py -> pipeline.py
       -> pretokenization.py -> tokenizer.py -> prepare_data.py
       -> experiments/ -> tests/
```

## 3. 环境和第一次运行

```bash
git branch --show-current
```

应该在 `experiment` 分支。CPU 最小环境：

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy regex
PYTHON=$PWD/.venv/bin/python bash experiments/run_experiments.sh
```

脚本会生成小语料、训练 2048-token BPE、生成 `.npy` 数据、训练四组模型并写入：

```text
artifacts/training_curves.svg
results/ablation_table.csv
technical_note.md
results/*.jsonl
checkpoints/*.pt
```

## 4. 语言模型在学什么

给定 token 序列：

```text
x0, x1, x2, x3, x4
```

训练样本错开一位：

```text
input:  x0, x1, x2, x3
target: x1, x2, x3, x4
```

模型输出：

```text
logits:  [B, T, V]
targets: [B, T]
```

每个位置是一个 `V` 类分类问题。当前交叉熵位于 `training.py`：

```python
log_normalizer = torch.logsumexp(inputs, dim=-1)
correct = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
return (log_normalizer - correct).mean()
```

这里 `inputs` 已经是 `[B*T, V]`。公式是：

```text
loss = mean(logsumexp(logits) - logits[正确 token])
```

练习：自己用 `torch.logsumexp` 和 `gather` 重写 loss，再和 `torch.nn.functional.cross_entropy` 对比。

## 5. Tokenizer：字符串如何变成数字

模型不能直接处理字符串，所以：

```text
"The cat sat" -> [17, 42, 91]
```

### Byte-Level BPE

`pretokenization.py` 先从 256 个 byte token 开始，然后反复合并高频相邻 pair：

```text
byte A + byte B -> 新 token
```

`run_train_bpe` 的过程：

1. 创建 byte vocabulary。
2. 加入 special tokens。
3. 用 `PAT` 正则切 pre-token。
4. 把每个 pre-token 转成 byte 序列。
5. 统计 pair frequency。
6. 合并最高频 pair，直到达到词表大小。

训练结果是：

```python
vocab[token_id] = bytes
merges = [(left_bytes, right_bytes), ...]
```

为什么从 byte 开始？任意 UTF-8 文本都能被表示，不需要维护一个巨大的字符表。

`tokenizer.py` 的 `_encode_piece` 按训练时的 merge rank 重放合并；`decode` 则把 token 对应的 bytes 拼起来，再解码 UTF-8。

先写最小版本：

```python
def encode(text):
    return list(text.encode("utf-8"))

def decode(ids):
    return bytes(ids).decode("utf-8")
```

确认 `decode(encode("你好 GPT"))` 正确后，再加入 pair merge。BPE 本质上就是在 byte tokenizer 上增加高频片段。

### `prepare_data.py`

`encode_file` 扫描两遍：第一遍计算 token 数，第二遍写入精确大小的 `.npy` memmap。输出是一维 token 数组：

```text
tokens: [N], dtype=uint16
```

如果词表小于 65536，`uint16` 足够保存 token ID。

## 6. Batch：从 token stream 取窗口

`training.py/get_batch` 随机选一个起点：

```python
starts = torch.randint(len(dataset) - context_length, (batch_size,))
```

若 `start=5`、`context_length=4`：

```text
input:  dataset[5:9]  = [5,6,7,8]
target: dataset[6:10] = [6,7,8,9]
```

最终：

```text
inputs:  [B, T]
targets: [B, T]
```

先自己用 NumPy 实现一个窗口，再和 `get_batch` 比较。必须理解 target 永远是 input 向右移动一位。

## 7. Embedding 和参数量

输入 token IDs：`[B,T]`。Embedding 矩阵：`[V,D]`。索引后：

```text
[B,T] -> [B,T,D]
```

当前代码：

```python
self.token_embeddings = nn.Embedding(vocab_size, d_model)
x = self.token_embeddings(token_ids)
```

Embedding 的手写版本只有一行：

```python
def embedding(weights, token_ids):
    return weights[token_ids]
```

`lm_head` 把 `[B,T,D]` 映射为 `[B,T,V]`。当前没有共享输入 embedding 和输出 head，因此两者各有 `V*D` 参数。

打印参数量：

```python
sum(p.numel() for p in model.parameters())
```

当前实验配置是：`V=4096,D=320,layers=4,heads=4,d_ff=1536`，总参数约 `10.16M`。

## 8. RMSNorm

对每个位置的 feature vector `x`：

```text
rms(x) = sqrt(mean(x^2) + eps)
RMSNorm(x) = x / rms(x) * learnable_weight
```

当前实现：

```python
values = x.float()
output = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
return (output * self.weight.float()).to(x.dtype)
```

`-1` 表示最后一个 feature 维；`[B,T,D]` 中每个位置单独归一化。先用 float32 计算统计量，是为了 fp16/bf16 数值稳定。

与 LayerNorm 相比，RMSNorm 不减均值，只调整尺度。

## 9. Attention

输入：`x:[B,T,D]`。三个投影：

```text
Q = xWq, K = xWk, V = xWv
```

当前四个 projection：

```python
self.q_proj = nn.Linear(d_model, d_model, bias=False)
self.k_proj = nn.Linear(d_model, d_model, bias=False)
self.v_proj = nn.Linear(d_model, d_model, bias=False)
self.output_proj = nn.Linear(d_model, d_model, bias=False)
```

若 `D=320,H=4`，每个 head dimension 是 `80`。reshape：

```text
[B,T,D] -> [B,H,T,D/H]
```

Attention 公式：

```text
scores = QK^T / sqrt(head_dim)
weights = softmax(scores)
output = weights V
```

形状：

```text
Q:      [B,H,T,d]
K^T:    [B,H,d,T]
scores: [B,H,T,T]
V:      [B,H,T,d]
```

因果语言模型必须禁止位置 `t` 读取 `t+1` 之后的信息。当前代码使用：

```python
torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

它等价于给 score 的上三角填 `-inf`，但通常更快。

### RoPE

RoPE 旋转 Q/K 的偶数、奇数 feature：

```text
even' = even*cos(angle) - odd*sin(angle)
odd'  = even*sin(angle) + odd*cos(angle)
```

`gpt.py` 在初始化时预计算 `cos` 和 `sin`，并用 `register_buffer` 保存。它们不是可学习参数，但应该跟着模型移动设备。

实现 attention 时，建议顺序：

1. 线性投影。
2. reshape 成 heads。
3. 对 Q/K 应用 RoPE。
4. causal attention。
5. transpose 回 `[B,T,D]`。
6. output projection。

练习：先写单头 `Q @ K.T`，再增加 head 维度；不要一开始就写完整 GPT。

## 10. SwiGLU

当前 FFN：

```text
SwiGLU(x) = W2( SiLU(W1x) * W3x )
SiLU(x) = x * sigmoid(x)
```

矩阵形状：

```text
W1: [d_ff,d_model]
W3: [d_ff,d_model]
W2: [d_model,d_ff]
```

输入 `[B,T,D]` 的中间形状：

```text
w1(x): [B,T,d_ff]
w3(x): [B,T,d_ff]
乘法:  [B,T,d_ff]
w2:    [B,T,D]
```

手写检查：

```python
hidden = torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)
y = hidden @ w2.T
```

## 11. Transformer Block

当前 baseline 是 Pre-Norm：

```text
x
 -> RMSNorm -> Attention -> x + attention
 -> RMSNorm -> SwiGLU   -> x + feed_forward
```

对应代码：

```python
attention = self.attn(self.ln1(x))
x = x + attention
normalized = self.ln2(x)
feed_forward = ...
return x + feed_forward
```

Residual 的含义是 `output=input+update(input)`。它给信息和梯度提供直接通道。当前 `no_residual` 消融用来观察没有这条路径时的影响。

Post-Norm 则是：

```text
x = RMSNorm(x + Attention(x))
x = RMSNorm(x + SwiGLU(x))
```

当前用 `--norm-position post` 切换。先写一个只有 `x + Linear(Norm(x))` 的小 block，确认 shape，再加入 attention 和 SwiGLU。

## 12. GPTLanguageModel

`gpt.py` 的初始化创建：

```python
token_embeddings
layers = nn.ModuleList([...])
ln_final
lm_head
```

forward：

```python
x = self.token_embeddings(token_ids)
for layer in self.layers:
    x = layer(x)
return self.lm_head(self.ln_final(x))
```

形状流：

```text
token_ids [B,T]
embedding [B,T,D]
每个 block [B,T,D]
lm_head [B,T,V]
```

如果 `T > context_length`，代码会抛错。生成时 `pipeline.generate` 只保留最近窗口。

初始化使用截断正态 Linear 权重，Embedding 用标准差 1 的正态。初始化会影响初期 logits 和梯度，因此也可以作为单变量实验。

## 13. 训练循环

`train.py` 的核心只有几步：

```python
inputs, targets = get_batch(...)
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
loss.backward()
gradient_clipping(model.parameters(), args.clip)
optimizer.step()
```

验证时使用：

```python
model.eval()
with torch.no_grad():
    ...
model.train()
```

训练日志包含：

```text
step, train_loss, valid_loss, lr, tokens_seen
```

固定 `random`、NumPy、PyTorch seed，才能把 ablation 的差异尽量归因于结构变量，而不是随机初始化。

### AdamW 直觉

SGD 是：`parameter -= lr * gradient`。

Adam 保存一阶、二阶移动平均：

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
```

再做 bias correction，然后按 `m_hat/(sqrt(v_hat)+eps)` 更新。AdamW 把 weight decay 独立乘到参数上。

`training.py` 有课程测试使用的完整 `AdamW`；`train.py` 有一个简化 `FastAdamW`，用于避开某些 CPU 环境的额外导入开销。正式长训练时应保存完整 optimizer state，才能可靠断点续训。

### 学习率

`cosine_lr` 先 warmup：

```text
lr = max_lr * step / warmup_steps
```

之后 cosine decay 到 `min_lr`。warmup 可以减少训练刚开始时的梯度爆炸风险。

## 14. 生成

`pipeline.py/generate` 的循环：

1. 运行当前 context。
2. 取最后位置 logits。
3. 除以 temperature。
4. 可选 top-k。
5. softmax。
6. multinomial 采样一个 token。
7. 拼接并重复。

```python
context = generated[:, -model.context_length:]
logits = model(context)
next_logits = logits[:, -1, :] / temperature
probabilities = torch.softmax(next_logits, dim=-1)
next_token = torch.multinomial(probabilities, 1)
generated = torch.cat((generated, next_token), dim=1)
```

`temperature<1` 更保守，`temperature>1` 更随机；top-k 只在最高 k 个 token 中采样。未训练模型生成没有意义，必须先加载 checkpoint，再用 tokenizer decode。

## 15. 从零重写的顺序

不要一次写完整 GPT。按以下顺序，每一步都写 shape test：

1. Linear：`[...,d_in] -> [...,d_out]`。
2. Embedding：`weights[token_ids]`。
3. RMSNorm：最后一维归一化。
4. 单头 causal attention。
5. Multi-head reshape 和合并。
6. RoPE。
7. SwiGLU。
8. Pre-Norm residual block。
9. embedding + blocks + final norm + lm head。
10. next-token loss。
11. AdamW 和训练循环。
12. checkpoint、sampling、decode。

每一步的输入输出 shape 要写在纸上。能写清楚 shape，通常就能定位 80% 的 Transformer bug。

## 16. 如何读测试

推荐顺序：

```bash
uv run pytest tests/test_data.py -q
uv run pytest tests/test_optimizer.py -q
uv run pytest tests/test_tokenizer.py -q
uv run pytest tests/test_model.py -q
```

`tests/_snapshots` 存了参考输出。测试会用固定输入和固定权重调用你的函数，再与 snapshot 比较。因此 shape 正确不代表计算正确。

常见错误：

- `nn.Linear` 权重是 `[d_out,d_in]`，所以手写是 `x @ weight.T`。
- causal mask 必须让位置 `i` 看到 `<=i`，不能看到 `>i`。
- attention reshape 前后要确认 head 维度没有错位。
- target 必须是 input 向右移动一位。
- tokenizer decode 需要拼 bytes 后再 UTF-8 decode。

## 17. 如何自己训练

最小命令：

```bash
.venv/bin/python -m cs336_basics.train \
  --train data_small/train.npy \
  --valid data_small/valid.npy \
  --vocab-size 4096 \
  --context-length 32 \
  --d-model 320 --layers 4 --heads 4 --d-ff 1536 \
  --steps 24 --eval-every 4 --batch-size 2 \
  --device cpu --threads 1 \
  --checkpoint checkpoints/my_run.pt \
  --log-path results/my_run.jsonl
```

先用很小数据验证 loss 会下降，再增加数据或 steps。不要一上来训练大语料；先证明模型能过拟合 100 个 token。

如果换真实 TinyStories：

1. 准备 train/valid 文本。
2. 用 `run_train_bpe` 训练 tokenizer。
3. pickle 保存 `vocab` 和 `merges`。
4. 用 `prepare_data.py` 生成 `.npy`。
5. 用 `train.py` 指定模型和训练超参数。

## 18. 如何设计消融实验

一次只动一个主要变量。当前四组：

| Run | 唯一结构变化 |
| --- | --- |
| baseline | Pre-Norm、保留 residual、context 32 |
| no_residual | residual 关闭 |
| post_norm | 改为 Post-Norm |
| context_64 | context 32 改为 64 |

必须记录 parameter count、train/validation loss、learning rate、tokens seen、wall-clock time。

Context 实验要特别小心：context 64 在相同 steps、相同 batch 下每步看到两倍 tokens。因此严格比较应该固定 `tokens_seen`，而不是只固定 steps。

可以自己增加：width、depth、d_ff、learning rate、GELU vs SwiGLU、RMSNorm vs LayerNorm、RoPE vs learned position embedding、初始化方式。

## 19. 当前实验结果怎么解释

当前表在 `results/ablation_table.csv`：

```text
baseline    valid_loss 3.3888
no_residual valid_loss 5.6388
post_norm   valid_loss 3.6200
context_64  valid_loss 2.6071
```

合理结论：residual 对短训练很重要；Pre-Norm 优于 Post-Norm；context 64 看起来最好，但它同时处理了两倍 tokens，因此不能单独证明长 context 天然更好。小语料和 24 steps 只适合做结构 sanity check，不适合声明 benchmark 结论。

## 20. 练习路线

### 初级

1. 手写 Linear、Embedding、softmax、cross entropy。
2. 手写 `get_batch`。
3. 给每个模块打印 shape。
4. 让模型在 20 个 token 上过拟合。

### 中级

1. 不调用 SDPA，手写 causal attention。
2. 手写 Multi-Head reshape。
3. 写 LayerNorm 并和 RMSNorm 对比。
4. 给生成增加 greedy decoding。
5. 实现 `<|endoftext|>` 停止生成。

### 高级

1. 实现 weight tying。
2. 保存完整 optimizer checkpoint 并支持 resume。
3. 做 equal-tokens-seen 的 context ablation。
4. 用真实 TinyStories 子集替换小语料。
5. 加入 dropout、混合精度、GPU 和梯度累积。

## 21. 最终自检清单

```text
[ ] 我能说出每个模块的输入输出 shape。
[ ] 我能解释 Q/K/V 和 causal mask。
[ ] 我能手写 RMSNorm、SwiGLU 和 cross entropy。
[ ] 我知道 input 和 target 为什么错开一位。
[ ] 我能从 token IDs 生成 logits。
[ ] 我能让一个极小数据集过拟合。
[ ] 我能保存并加载 checkpoint。
[ ] 我能从 prompt 逐 token 生成。
[ ] 我能设计一次只改一个变量的 ablation。
[ ] 我能解释结果的不确定性。
```

真正掌握 GPT 的标志，不是记住类名，而是能在纸上写出主要公式、张量形状和训练循环，并能从一个最小版本逐步扩展到当前代码。
