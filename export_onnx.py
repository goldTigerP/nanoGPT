"""
将 nanoGPT 的 .pth 模型文件转换为 ONNX 格式。

用法:
    python export_onnx.py --pth_path out/model.pth --config_path out/model_config.pkl --output_path out/model.onnx
    
    # 也可以从 ckpt.pt checkpoint 文件转换
    python export_onnx.py --ckpt_path out/ckpt.pt --output_path out/model.onnx

    # 指定序列长度和设备
    python export_onnx.py --pth_path out/model.pth --config_path out/model_config.pkl --seq_len 128 --device cpu
"""

import os
import argparse
import pickle

import torch
import torch.nn as nn

from model import GPTConfig, GPT


def load_model_from_pth(pth_path: str, config_path: str, device: str = 'cpu') -> GPT:
    """从 .pth 权重文件 + model_config.pkl 加载模型"""
    # 加载模型配置
    with open(config_path, 'rb') as f:
        model_args = pickle.load(f)
    print(f"模型配置: {model_args}")

    # 构建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # 加载权重
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"已从 {pth_path} 加载模型权重")

    return model


def load_model_from_ckpt(ckpt_path: str, device: str = 'cpu') -> GPT:
    """从 ckpt.pt checkpoint 文件加载模型"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    print(f"模型配置: {model_args}")

    # 构建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # 处理可能的 _orig_mod. 前缀 (torch.compile 产生的)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    print(f"已从 {ckpt_path} 加载 checkpoint (iter {checkpoint.get('iter_num', '?')})")

    return model


def export_to_onnx(model: GPT, output_path: str, seq_len: int = 64,
                   device: str = 'cpu', opset_version: int = 17):
    """将 GPT 模型导出为 ONNX 格式"""

    model = model.to(device)
    model.eval()

    # 禁用 flash attention 以兼容 ONNX 导出
    # ONNX 不支持 scaled_dot_product_attention，需要回退到手动实现
    for block in model.transformer.h:
        block.attn.flash = False
        # 如果之前使用 flash attention，需要注册 causal mask buffer
        if not hasattr(block.attn, 'bias') or block.attn.bias is None:
            block_size = model.config.block_size
            bias = torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            )
            block.attn.register_buffer("bias", bias)

    # 构造虚拟输入 (batch_size=1, seq_len)
    dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len),
                                dtype=torch.long, device=device)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    print(f"正在导出 ONNX 模型...")
    print(f"  序列长度: {seq_len}")
    print(f"  输出路径: {output_path}")
    print(f"  opset 版本: {opset_version}")

    # 导出 ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),                    # 模型输入 (仅 idx，不传 targets)
            output_path,
            export_params=True,                # 导出训练好的参数
            opset_version=opset_version,
            do_constant_folding=True,          # 常量折叠优化
            input_names=['input_ids'],         # 输入名称
            output_names=['logits'],           # 输出名称 (推理模式只有 logits)
            dynamic_axes={                     # 支持动态 batch 和序列长度
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits':    {0: 'batch_size', 1: 'sequence_length'},
            },
        )

    # 验证导出的模型
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ ONNX 模型已导出成功!")
    print(f"   文件路径: {output_path}")
    print(f"   文件大小: {file_size_mb:.2f} MB")

    # 尝试用 onnxruntime 验证
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        test_input = dummy_input.cpu().numpy()
        ort_output = session.run(None, {input_name: test_input})

        # 与 PyTorch 输出对比
        with torch.no_grad():
            pt_output, _ = model(dummy_input)
        pt_output_np = pt_output.cpu().numpy()

        max_diff = abs(ort_output[0] - pt_output_np).max()
        print(f"\n🔍 验证结果:")
        print(f"   PyTorch 输出形状: {pt_output_np.shape}")
        print(f"   ONNX 输出形状:    {ort_output[0].shape}")
        print(f"   最大误差:         {max_diff:.6e}")
        if max_diff < 1e-3:
            print(f"   ✅ 精度验证通过!")
        else:
            print(f"   ⚠️  精度差异较大，请检查模型")
    except ImportError:
        print("\n💡 提示: 安装 onnxruntime 可以验证导出结果:")
        print("   pip install onnxruntime")


def main():
    parser = argparse.ArgumentParser(description='将 nanoGPT .pth 模型转换为 ONNX 格式')

    # 输入方式1: pth + config
    parser.add_argument('--pth_path', type=str, default=None,
                        help='.pth 模型权重文件路径')
    parser.add_argument('--config_path', type=str, default=None,
                        help='model_config.pkl 配置文件路径')

    # 输入方式2: ckpt checkpoint
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='ckpt.pt checkpoint 文件路径')

    # 输出
    parser.add_argument('--output_path', type=str, default='out/model.onnx',
                        help='ONNX 输出文件路径 (默认: out/model.onnx)')

    # 参数
    parser.add_argument('--seq_len', type=int, default=64,
                        help='导出时使用的序列长度 (默认: 64)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (默认: cpu)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset 版本 (默认: 17)')

    args = parser.parse_args()

    # 参数校验
    if args.ckpt_path is not None:
        # 从 checkpoint 加载
        model = load_model_from_ckpt(args.ckpt_path, args.device)
    elif args.pth_path is not None and args.config_path is not None:
        # 从 pth + config 加载
        model = load_model_from_pth(args.pth_path, args.config_path, args.device)
    else:
        # 尝试自动检测常见路径
        default_paths = [
            ('out/model.pth', 'out/model_config.pkl'),
            ('out-hongloumeng/model.pth', 'out-hongloumeng/model_config.pkl'),
            ('out-hongloumeng-fast/model.pth', 'out-hongloumeng-fast/model_config.pkl'),
        ]
        default_ckpts = [
            'out/ckpt.pt',
            'out-hongloumeng/ckpt.pt',
            'out-hongloumeng-fast/ckpt.pt',
        ]

        model = None
        # 优先查找 pth 文件
        for pth, cfg in default_paths:
            if os.path.exists(pth) and os.path.exists(cfg):
                print(f"自动检测到模型文件: {pth}")
                model = load_model_from_pth(pth, cfg, args.device)
                break

        # 如果没找到 pth，查找 checkpoint
        if model is None:
            for ckpt in default_ckpts:
                if os.path.exists(ckpt):
                    print(f"自动检测到 checkpoint: {ckpt}")
                    model = load_model_from_ckpt(ckpt, args.device)
                    break

        if model is None:
            parser.error(
                "未找到模型文件。请通过以下方式之一指定:\n"
                "  --pth_path <path> --config_path <path>\n"
                "  --ckpt_path <path>"
            )

    # 导出 ONNX
    export_to_onnx(model, args.output_path, args.seq_len, args.device, args.opset)


if __name__ == '__main__':
    main()
