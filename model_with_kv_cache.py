"""
nanoGPT with KV Cache implementation
为nanoGPT添加KV Cache支持，大幅提升推理速度
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class KVCache:
    """
    Key-Value Cache for efficient autoregressive generation
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, dtype=torch.float32, device='cpu'):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # 预分配cache内存 [batch_size, n_heads, seq_len, head_dim]
        self.k_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), 
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), 
                                   dtype=dtype, device=device)
        self.seq_len = 0  # 当前缓存的序列长度
    
    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV Cache并返回完整的key和value
        
        Args:
            k: 新的key tensor [batch_size, n_heads, seq_len, head_dim]
            v: 新的value tensor [batch_size, n_heads, seq_len, head_dim] 
            start_pos: 开始位置
            
        Returns:
            完整的key和value tensor
        """
        batch_size, n_heads, seq_len, head_dim = k.shape
        
        # 更新cache
        self.k_cache[:batch_size, :, start_pos:start_pos + seq_len] = k
        self.v_cache[:batch_size, :, start_pos:start_pos + seq_len] = v
        
        # 更新当前序列长度
        self.seq_len = max(self.seq_len, start_pos + seq_len)
        
        # 返回完整的key和value
        keys = self.k_cache[:batch_size, :, :self.seq_len]
        values = self.v_cache[:batch_size, :, :self.seq_len]
        
        return keys, values
    
    def reset(self):
        """重置cache"""
        self.seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

class CausalSelfAttentionWithCache(nn.Module):
    """
    支持KV Cache的因果自注意力层
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # 如果使用KV Cache
        if kv_cache is not None:
            # 更新cache并获取完整的key和value
            k, v = kv_cache.update(k, v, start_pos)
            # 注意：query只对应当前输入的token，但key和value包含所有历史token

        # causal self-attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Flash Attention自动处理因果掩码
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # 应用因果掩码
            seq_len = k.size(2)  # 完整序列长度（包括cache）
            query_len = q.size(2)  # 查询序列长度（当前输入）
            
            # 创建因果掩码：确保每个query位置只能看到它之前（包括自己）的key位置
            if kv_cache is not None:
                # 使用KV Cache时，需要特殊处理掩码
                # query的位置是[start_pos:start_pos+query_len]
                # key的位置是[0:seq_len]
                mask = torch.triu(torch.ones(query_len, seq_len, device=x.device), 
                                 diagonal=seq_len - start_pos - query_len + 1)
                att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))
            else:
                # 标准因果掩码
                att = att.masked_fill(self.bias[:,:,:query_len,:seq_len] == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T_q, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T_q, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, q.size(2), C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class BlockWithCache(nn.Module):
    """
    支持KV Cache的Transformer Block
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionWithCache(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        x = x + self.attn(self.ln_1(x), kv_cache, start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPTWithKVCache(nn.Module):
    """
    支持KV Cache的GPT模型
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BlockWithCache(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches: Optional[List[KVCache]] = None, start_pos: int = 0):
        device = idx.device
        b, t = idx.size()
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # Position embeddings
        if kv_caches is not None:
            # 使用KV Cache时，位置embedding需要考虑start_pos
            pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        pos_emb = self.transformer.wpe(pos) # position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for i, block in enumerate(self.transformer.h):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache, start_pos)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training mode
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode
            logits = self.lm_head(x[:, [-1], :]) if kv_caches is not None else self.lm_head(x)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate_with_cache(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        使用KV Cache的高效生成方法
        """
        batch_size = idx.size(0)
        device = idx.device
        
        # 初始化KV Cache
        kv_caches = []
        head_dim = self.config.n_embd // self.config.n_head
        for _ in range(self.config.n_layer):
            cache = KVCache(
                max_batch_size=batch_size,
                max_seq_len=self.config.block_size,
                n_heads=self.config.n_head,
                head_dim=head_dim,
                dtype=next(self.parameters()).dtype,
                device=device
            )
            kv_caches.append(cache)
        
        # Prefill阶段：处理初始序列
        seq_len = idx.size(1)
        if seq_len > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            seq_len = self.config.block_size
            
        logits, _ = self(idx, kv_caches=kv_caches, start_pos=0)
        start_pos = seq_len
        
        # 生成新token
        for _ in range(max_new_tokens):
            # 采样
            logits_last = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 前向传播新token（这里只需要处理一个token！）
            if start_pos < self.config.block_size:
                logits, _ = self(idx_next, kv_caches=kv_caches, start_pos=start_pos)
                start_pos += 1
            else:
                # 如果超过最大长度，需要重新开始（这里简化处理）
                break
            
            # 拼接结果
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 限制输出长度
            if idx.size(1) > self.config.block_size:
                idx = idx[:, -self.config.block_size:]

        return idx

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """
        生成文本，支持选择是否使用KV Cache
        """
        if use_cache:
            return self.generate_with_cache(idx, max_new_tokens, temperature, top_k)
        else:
            # 原始的生成方法（无cache）
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx


# 使用示例和性能对比
if __name__ == "__main__":
    import time
    
    # 配置
    config = GPTConfig(
        block_size=512,
        vocab_size=4656,  # 红楼梦词汇表大小
        n_layer=6,
        n_head=6,
        n_embd=384
    )
    
    # 创建模型
    model = GPTWithKVCache(config)
    model.eval()
    
    # 测试数据
    batch_size = 1
    seq_len = 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("=" * 60)
    print("KV Cache性能对比测试")
    print("=" * 60)
    
    # 测试无cache生成
    start_time = time.time()
    with torch.no_grad():
        output_no_cache = model.generate(idx.clone(), max_new_tokens=50, use_cache=False)
    time_no_cache = time.time() - start_time
    
    # 测试有cache生成
    start_time = time.time()
    with torch.no_grad():
        output_with_cache = model.generate(idx.clone(), max_new_tokens=50, use_cache=True)
    time_with_cache = time.time() - start_time
    
    print(f"无KV Cache生成时间: {time_no_cache:.3f}s")
    print(f"有KV Cache生成时间: {time_with_cache:.3f}s")
    print(f"加速比: {time_no_cache/time_with_cache:.2f}x")
    print(f"输出序列长度: {output_with_cache.size(1)}")
    
    # 验证输出一致性（前面部分应该相同）
    print(f"输出一致性检查: {torch.equal(output_no_cache[:, :seq_len+10], output_with_cache[:, :seq_len+10])}")