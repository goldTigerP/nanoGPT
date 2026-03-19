#!/usr/bin/env python3
"""
红楼梦模型文本生成测试脚本
"""

import os
import sys
import torch
import pickle
from contextlib import nullcontext

# 添加nanoGPT目录到路径
sys.path.append('/home/pengjh5/PRJ/ml/nanoGPT')
from model import GPTConfig, GPT

def load_model():
    """加载训练好的红楼梦模型"""
    
    # 设置设备
    device = 'cpu'
    dtype = torch.float32
    device_type = 'cpu'
    ctx = nullcontext()
    
    # 加载检查点
    ckpt_path = 'out-hongloumeng-fast/ckpt.pt'
    print(f"正在加载模型: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 加载配置
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # 加载模型权重
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"模型加载完成！参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载词汇表
    meta_path = 'data/chinese_literature/meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    def encode(s):
        return [stoi[c] for c in s if c in stoi]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    return model, encode, decode, device, dtype, ctx

def generate_text(model, encode, decode, device, dtype, ctx, 
                 start_text="红楼梦", max_new_tokens=200, temperature=0.8, top_k=200):
    """生成红楼梦风格的文本"""
    
    # 编码起始文本
    start_ids = encode(start_text)
    if not start_ids:
        print("警告：起始文本包含未知字符，使用默认起始")
        start_ids = encode("第")
        if not start_ids:
            start_ids = [0]  # 使用第一个token作为默认
    
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    print(f"起始文本: '{start_text}'")
    print(f"开始生成 {max_new_tokens} 个字符...")
    print("-" * 50)
    
    # 生成文本
    with torch.no_grad():
        with ctx:
            for k in range(max_new_tokens):
                # 如果序列太长，截断
                if x.size(1) > 2048:
                    x = x[:, -1024:]
                
                # 前向传播
                logits, _ = model(x)
                logits = logits[:, -1, :] / temperature
                
                # 可选的top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # 采样
                probs = torch.nn.functional.softmax(logits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1)
                
                # 添加到序列
                x = torch.cat((x, ix), dim=1)
                
                # 实时显示生成的字符
                new_char = decode(ix[0].tolist())
                print(new_char, end='', flush=True)
    
    print("\n" + "-" * 50)
    
    # 返回完整生成的文本
    generated = decode(x[0].tolist())
    return generated

def main():
    """主函数"""
    print("🏮 红楼梦语言模型文本生成器")
    print("=" * 50)
    
    # 加载模型
    try:
        model, encode, decode, device, dtype, ctx = load_model()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print("\n模型加载成功！现在可以生成文本了。")
    print("输入 'quit' 退出程序")
    print("=" * 50)
    
    while True:
        # 获取用户输入
        start_text = input("\n请输入起始文本 (默认'红楼梦'): ").strip()
        if start_text.lower() == 'quit':
            break
        
        if not start_text:
            start_text = "红楼梦"
        
        # 生成参数
        try:
            max_tokens = input("生成长度 (默认200): ").strip()
            max_tokens = int(max_tokens) if max_tokens else 200
            max_tokens = min(max_tokens, 1000)  # 限制最大长度
            
            temp = input("温度参数 (默认0.8): ").strip()
            temp = float(temp) if temp else 0.8
            temp = max(0.1, min(temp, 2.0))  # 限制温度范围
            
        except ValueError:
            print("参数输入错误，使用默认值")
            max_tokens = 200
            temp = 0.8
        
        print(f"\n生成参数: 长度={max_tokens}, 温度={temp}")
        
        # 生成文本
        try:
            generated_text = generate_text(
                model, encode, decode, device, dtype, ctx,
                start_text=start_text, 
                max_new_tokens=max_tokens, 
                temperature=temp
            )
            
            print(f"\n\n完整生成文本:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ 文本生成失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()