"""
中文文学作品文本生成脚本
基于训练好的中文GPT模型生成文本
"""
import os
import pickle
import torch
from model import GPTConfig, GPT

def generate_chinese_text(
    out_dir='out-chinese-literature',
    start='红楼梦',
    num_samples=5,
    max_new_tokens=200,
    temperature=0.8,
    top_k=200,
    seed=1337,
    device='cuda',
    dtype='bfloat16'
):
    """
    生成中文文本
    
    Args:
        out_dir: 模型检查点目录
        start: 开始文本
        num_samples: 生成样本数量
        max_new_tokens: 最大新token数
        temperature: 温度参数，控制随机性
        top_k: top-k采样
        seed: 随机种子
        device: 设备
        dtype: 数据类型
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # 加载模型
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    
    # 加载元数据
    meta_path = os.path.join('data', 'chinese_literature', 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("没有找到meta.pkl文件，使用tiktoken编码")
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    # 编码开始文本
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    # 生成文本
    print(f"开始文本: {start}")
    print("=" * 50)
    
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated_text = decode(y[0].tolist())
                print(f"样本 {k+1}:")
                print(generated_text)
                print('-' * 30)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成中文文学作品文本')
    parser.add_argument('--out_dir', type=str, default='out-chinese-literature', help='模型输出目录')
    parser.add_argument('--start', type=str, default='红楼梦', help='开始文本')
    parser.add_argument('--num_samples', type=int, default=5, help='生成样本数量')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='最大新token数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度参数')
    parser.add_argument('--top_k', type=int, default=200, help='top-k采样')
    parser.add_argument('--seed', type=int, default=1337, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='数据类型')
    parser.add_argument('--compile', action='store_true', help='编译模型')
    
    args = parser.parse_args()
    
    generate_chinese_text(**vars(args))