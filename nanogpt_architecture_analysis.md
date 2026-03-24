# nanoGPT模型结构详解与KV Cache实现

## 1. nanoGPT整体架构

nanoGPT是一个经典的GPT架构实现，结构清晰简洁：

```
GPT模型
├── Token Embedding (wte): 词汇表 -> 嵌入向量
├── Position Embedding (wpe): 位置 -> 嵌入向量 
├── Transformer Blocks (h): N层Transformer块
│   ├── Layer Norm 1 (ln_1)
│   ├── Causal Self-Attention (attn)
│   │   ├── Query/Key/Value投影 (c_attn)
│   │   ├── Multi-Head Attention计算
│   │   └── 输出投影 (c_proj)
│   ├── 残差连接
│   ├── Layer Norm 2 (ln_2) 
│   ├── MLP前馈网络
│   │   ├── 扩展层 (c_fc): embed_dim -> 4*embed_dim
│   │   ├── GELU激活函数
│   │   └── 压缩层 (c_proj): 4*embed_dim -> embed_dim
│   └── 残差连接
├── 最终Layer Norm (ln_f)
└── 语言模型头 (lm_head): 嵌入 -> 词汇表概率
```

## 2. 核心组件详解

### 2.1 Causal Self-Attention机制
- **QKV投影**: 一个线性层同时计算Q、K、V，然后split
- **多头注意力**: 将嵌入维度分割到多个头
- **因果掩码**: 确保只能看到当前位置之前的token
- **Flash Attention**: 支持PyTorch 2.0+的优化实现

### 2.2 MLP前馈网络
- 经典的两层线性变换，中间使用GELU激活
- 隐藏层维度是嵌入维度的4倍（标准GPT配置）

### 2.3 权重共享
- Token embedding和语言模型头共享权重（Weight Tying）
- 减少参数数量，提高训练效率

## 3. 当前生成方法的性能问题

现在的`generate`方法存在严重的性能问题：

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 每次都要重新计算所有token的attention！
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)  # 完整前向传播
        # ... 采样逻辑
```

**问题**: 
- 每生成一个token都要重新计算整个序列的attention
- 时间复杂度是O(n²)，序列越长越慢
- 大量重复计算，浪费资源

## 4. KV Cache原理

### 4.1 核心思想
在自回归生成中，对于已生成的token，它们的Key和Value在后续步骤中不会改变。
我们可以缓存这些Key和Value，避免重复计算。

### 4.2 数学原理
对于位置i的Query，它需要与位置0到i的所有Key计算attention：
```
Attention(Q_i) = softmax(Q_i @ [K_0, K_1, ..., K_i]) @ [V_0, V_1, ..., V_i]
```

在生成第i+1个token时：
- K_0到K_i已经计算过，可以直接使用缓存
- 只需要计算新的K_{i+1}和V_{i+1}

### 4.3 复杂度优化
- **无KV Cache**: O(n²) - 每步重计算所有token
- **有KV Cache**: O(n) - 每步只计算新token

## 5. KV Cache实现

让我们实现一个支持KV Cache的版本：