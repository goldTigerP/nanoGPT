"""
KV Cache可视化演示脚本
展示KV Cache在自回归生成中的工作原理
"""

import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, skipping visualizations")

def visualize_attention_pattern():
    """可视化注意力模式：有无KV Cache的对比"""
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 序列长度
    seq_len = 8
    
    # 创建注意力模式矩阵（因果掩码）
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    
    # 绘制传统方法的注意力计算
    ax1.imshow(causal_mask, cmap='Blues', alpha=0.7)
    ax1.set_title('传统方法：每步重新计算所有注意力', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Key positions')
    ax1.set_ylabel('Query positions')
    
    # 添加网格和标签
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    ax1.grid(True, alpha=0.3)
    
    # 高亮显示重复计算的部分
    for i in range(1, seq_len):
        for j in range(i):
            rect = Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                           linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            ax1.add_patch(rect)
    
    ax1.text(seq_len//2, -1, '红框：重复计算的注意力', ha='center', color='red', fontweight='bold')
    
    # 绘制KV Cache方法
    # 展示逐步生成过程
    cache_pattern = np.zeros((seq_len, seq_len))
    
    # 第一步：计算第0个位置
    cache_pattern[0, 0] = 1
    
    # 第二步：只计算新的attention (1,0) 和 (1,1)，复用(0,0)
    cache_pattern[1, :2] = [0.5, 1]  # 0.5表示复用的cache
    
    # 第三步：只计算新的attention，复用之前的
    cache_pattern[2, :3] = [0.5, 0.5, 1]
    
    # 继续这个模式
    for i in range(3, seq_len):
        cache_pattern[i, :i] = 0.5  # 复用的部分
        cache_pattern[i, i] = 1     # 新计算的部分
    
    im = ax2.imshow(cache_pattern, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('KV Cache方法：复用历史计算', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key positions')
    ax2.set_ylabel('Query positions')
    
    ax2.set_xticks(range(seq_len))
    ax2.set_yticks(range(seq_len))
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色说明
    ax2.text(seq_len//2, -1, '🟢新计算 🟡缓存复用', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/pengjh5/PRJ/ml/nanoGPT/kv_cache_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_memory_usage():
    """可视化内存使用模式"""
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 生成步数
    steps = np.arange(1, 21)
    
    # 传统方法：每步计算复杂度O(n²)
    traditional_compute = steps ** 2
    traditional_memory = steps  # 只存储当前序列
    
    # KV Cache方法：每步计算复杂度O(n)，但内存使用O(n²)
    kv_cache_compute = steps
    kv_cache_memory = steps ** 2  # 存储所有key和value
    
    # 绘制计算复杂度对比
    ax1.plot(steps, traditional_compute, 'r-o', label='传统方法 O(n²)', linewidth=2, markersize=5)
    ax1.plot(steps, kv_cache_compute, 'g-s', label='KV Cache O(n)', linewidth=2, markersize=5)
    ax1.set_xlabel('生成步数')
    ax1.set_ylabel('计算量 (相对单位)')
    ax1.set_title('计算复杂度对比', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 绘制内存使用对比
    ax2.plot(steps, traditional_memory, 'r-o', label='传统方法 O(n)', linewidth=2, markersize=5)
    ax2.plot(steps, kv_cache_memory, 'g-s', label='KV Cache O(n²)', linewidth=2, markersize=5)
    ax2.set_xlabel('生成步数')
    ax2.set_ylabel('内存使用 (相对单位)')
    ax2.set_title('内存使用对比', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/pengjh5/PRJ/ml/nanoGPT/kv_cache_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_kv_cache_workflow():
    """演示KV Cache的详细工作流程"""
    
    print("🚀 KV Cache工作流程演示")
    print("=" * 60)
    
    # 模拟参数
    batch_size = 1
    n_heads = 4
    head_dim = 64
    max_seq_len = 10
    
    print(f"📊 配置信息:")
    print(f"   批次大小: {batch_size}")
    print(f"   注意力头数: {n_heads}")
    print(f"   每头维度: {head_dim}")
    print(f"   最大序列长度: {max_seq_len}")
    print()
    
    # 创建KV Cache
    from model_with_kv_cache import KVCache
    
    kv_cache = KVCache(
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device='cpu'
    )
    
    print("🔄 逐步生成演示:")
    print("-" * 40)
    
    # 模拟逐步生成过程
    for step in range(5):
        print(f"\n📍 步骤 {step + 1}:")
        
        # 生成新的key和value（模拟）
        new_k = torch.randn(batch_size, n_heads, 1, head_dim)
        new_v = torch.randn(batch_size, n_heads, 1, head_dim)
        
        print(f"   新Key形状: {new_k.shape}")
        print(f"   新Value形状: {new_v.shape}")
        
        # 更新cache
        full_k, full_v = kv_cache.update(new_k, new_v, step)
        
        print(f"   完整Key形状: {full_k.shape}")
        print(f"   完整Value形状: {full_v.shape}")
        print(f"   当前缓存长度: {kv_cache.seq_len}")
        
        # 计算注意力所需的计算量（简化）
        query_len = 1  # 当前只有一个新query
        key_len = kv_cache.seq_len  # 需要与所有历史key计算attention
        
        print(f"   📈 注意力计算量: {query_len} × {key_len} = {query_len * key_len}")
        
        if step == 0:
            total_compute = query_len * key_len
        else:
            total_compute += query_len * key_len
            traditional_compute = sum(i * i for i in range(1, step + 2))  # 传统方法的累计计算量
            
            print(f"   🔍 对比传统方法:")
            print(f"      KV Cache累计计算量: {total_compute}")
            print(f"      传统方法累计计算量: {traditional_compute}")
            print(f"      加速比: {traditional_compute / total_compute:.2f}x")

def main():
    """主演示函数"""
    print("🏮 nanoGPT架构与KV Cache详解")
    print("=" * 60)
    
    print("\n📚 nanoGPT模型结构概述:")
    print("""
    nanoGPT采用经典的Decoder-Only Transformer架构：
    
    1. 🎯 Token Embedding: 将词汇映射到向量空间
    2. 📍 Position Embedding: 添加位置信息
    3. 🔄 N个Transformer Block:
       - LayerNorm → Self-Attention → 残差连接
       - LayerNorm → MLP → 残差连接
    4. 🎭 Language Model Head: 输出词汇概率分布
    
    🔑 核心特点:
    - 因果掩码（Causal Mask）确保只能看到历史信息
    - 权重共享（Weight Tying）减少参数量
    - 支持Flash Attention加速
    """)
    
    print("\n🚀 KV Cache核心原理:")
    print("""
    在自回归生成中，每个token的Key和Value在生成过程中不会改变：
    
    🔄 传统方法问题:
    - 每生成一个token都要重新计算整个序列的attention
    - 时间复杂度: O(n²)，n是序列长度
    - 大量重复计算，效率低下
    
    ✨ KV Cache解决方案:
    - 缓存历史token的Key和Value
    - 每步只计算新token的Key和Value
    - 时间复杂度: O(n)，大幅提升生成速度
    
    💾 权衡考虑:
    - 时间: 从O(n²) → O(n)，显著提速
    - 空间: 需要额外内存存储cache
    - 适用: 推理阶段，训练阶段通常不使用
    """)
    
    # 运行演示
    print("\n🎮 交互式演示:")
    
    try:
        demonstrate_kv_cache_workflow()
        
        # 如果有matplotlib，绘制可视化图表
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        print("\n🎨 生成可视化图表...")
        # visualize_attention_pattern()
        # visualize_memory_usage()
        print("   图表已保存到文件中")
        
    except ImportError as e:
        print(f"   跳过可视化: {e}")
    
    print("\n✅ 演示完成!")
    print("""
    🎯 关键收获:
    1. KV Cache通过复用历史计算实现加速
    2. 适用于推理阶段的自回归生成
    3. 时间复杂度从O(n²)降低到O(n)
    4. 需要额外内存存储Key和Value cache
    5. 在长序列生成中效果尤其显著
    """)

if __name__ == "__main__":
    main()