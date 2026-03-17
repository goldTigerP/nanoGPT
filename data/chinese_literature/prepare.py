"""
红楼梦数据准备脚本
专门处理红楼梦文本，提供可靠的数据源
"""
import os
import pickle
import numpy as np
import urllib.request
import urllib.error

def download_hongloumeng():
    """下载完整的红楼梦文本"""
    filename = 'hongloumeng_full.txt'
    
    # 多个可能的下载源
    urls = [
        # Project Gutenberg中文文本
        'https://www.gutenberg.org/files/9603/9603-0.txt',
        # 另一个公共源
        'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/红楼梦/红楼梦.txt',
        # 备用源
        'https://www.haodoo.net/?M=d&P=red:1'
    ]
    
    for i, url in enumerate(urls):
        try:
            print(f"尝试从源 {i+1} 下载红楼梦...")
            print(f"URL: {url}")
            
            # 设置请求头，模拟浏览器
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                
                # 尝试不同的编码
                for encoding in ['utf-8', 'gb2312', 'gbk', 'big5']:
                    try:
                        text = content.decode(encoding)
                        print(f"成功使用 {encoding} 编码解码")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"无法解码从 {url} 获取的内容")
                    continue
                
                # 检查是否包含中文内容
                chinese_chars = len([c for c in text[:1000] if '\u4e00' <= c <= '\u9fff'])
                if chinese_chars < 50:  # 如果前1000字符中中文字符少于50个，可能不是中文文本
                    print(f"从 {url} 获取的内容似乎不是中文文本")
                    continue
                
                # 保存文件
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"✅ 成功下载红楼梦！文件大小: {len(text):,} 字符")
                print(f"保存为: {filename}")
                return filename
                
        except Exception as e:
            print(f"从源 {i+1} 下载失败: {e}")
            continue
    
    print("❌ 所有下载源都失败了，将使用内嵌的示例文本")
    return None

def prepare_hongloumeng():
    """准备红楼梦训练数据"""
    
    print("🔄 开始准备红楼梦训练数据...")
    
    # 优先使用本地完整版红楼梦文本
    local_hongloumeng = '红楼梦_完整版.txt'
    
    if os.path.exists(local_hongloumeng):
        print(f"✅ 找到本地红楼梦完整版: {local_hongloumeng}")
        downloaded_file = local_hongloumeng
    else:
        # 如果本地文件不存在，尝试下载
        print("本地完整版不存在，尝试在线下载...")
        downloaded_file = download_hongloumeng()
    
    # 红楼梦内嵌文本（作为备用）
    hongloumeng_text = ""

    # 检查是否存在现有的文本文件
    existing_files = []
    
    # 首先检查本地完整版红楼梦
    if os.path.exists(local_hongloumeng):
        existing_files.append(local_hongloumeng)
        print(f"✅ 找到本地完整版红楼梦: {local_hongloumeng}")
    elif downloaded_file:
        existing_files.append(downloaded_file)
        print(f"✅ 使用下载的文件: {downloaded_file}")
    
    # 检查其他可能的完整版文件
    if os.path.exists('hongloumeng_complete.txt'):
        existing_files.append('hongloumeng_complete.txt')
        print("✅ 找到完整版红楼梦文件")
    
    # 然后检查其他可能的文件
    other_files = ['hongloumeng.txt', 'xiyouji.txt', 'shuihuzhuan.txt', 'sanguo.txt']
    for filename in other_files:
        if os.path.exists(filename):
            existing_files.append(filename)
    # 合并所有可用的文本
    all_text = ""
    
    # 合并所有可用的文本
    all_text = ""
    
    # 如果有现有文件，优先使用
    if existing_files:
        for filename in existing_files:
            try:
                print(f"📖 加载文件: {filename}")
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_text += content + "\n\n"
                    print(f"✅ 成功加载 {filename}，长度: {len(content):,} 字符")
            except Exception as e:
                print(f"❌ 读取 {filename} 失败: {e}")
    
    # 如果没有现有文件或文件太小，使用内嵌文本
    if len(all_text) < 10000:
        print("📝 文本内容不足，添加内嵌的红楼梦示例文本")
        all_text += hongloumeng_text

    # 如果内容较少，重复文本以增加训练数据量
    if len(all_text) < 50000:  # 如果文本少于5万字符
        repeat_times = max(1, 50000 // len(all_text))
        all_text = all_text * repeat_times
        print(f"文本内容较少，重复 {repeat_times} 次增加训练数据")

    print(f"\n总文本长度: {len(all_text):,} 字符")
    
    # 文本清洗
    print("清洗文本...")
    # 移除多余的空行和空格
    lines = [line.strip() for line in all_text.split('\n') if line.strip()]
    all_text = '\n'.join(lines)
    
    # 创建字符级别的词汇表
    print("创建字符级词汇表...")
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    print(f"词汇表大小: {vocab_size} 个独特字符")
    
    # 显示词汇表的组成
    chinese_chars = [c for c in chars if '\u4e00' <= c <= '\u9fff']
    punctuations = [c for c in chars if c in '，。！？；：""''（）【】『』《》〈〉「」']
    digits = [c for c in chars if c.isdigit()]
    others = [c for c in chars if c not in chinese_chars and c not in punctuations and c not in digits]
    
    print(f"  - 中文字符: {len(chinese_chars)} 个")
    print(f"  - 标点符号: {len(punctuations)} 个")
    print(f"  - 数字: {len(digits)} 个")
    print(f"  - 其他字符: {len(others)} 个 (包括字母、空格等)")
    print(f"词汇表示例: {''.join(chars[:50])}")
    
    # 创建字符到索引的映射
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # 编码整个文本
    print("编码文本...")
    data = np.array(encode(all_text), dtype=np.uint16)
    print(f"编码后数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    
    # 划分训练集和验证集
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    print(f"\n数据划分:")
    print(f"  训练集大小: {len(train_data):,} tokens ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  验证集大小: {len(val_data):,} tokens ({len(val_data)/len(data)*100:.1f}%)")
    
    # 保存数据
    print("保存数据文件...")
    train_data.tofile('train.bin')
    val_data.tofile('val.bin')
    
    # 保存元数据
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'chars': chars,
        'data_info': {
            'total_chars': len(all_text),
            'chinese_chars': len(chinese_chars),
            'unique_chars': vocab_size,
            'train_tokens': len(train_data),
            'val_tokens': len(val_data)
        }
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print("\n✅ 数据准备完成！")
    print("生成的文件:")
    print("  - train.bin: 训练数据")
    print("  - val.bin: 验证数据") 
    print("  - meta.pkl: 词汇表元数据")
    
    # 测试编码解码
    print(f"\n🧪 编码解码测试:")
    test_texts = ["红楼梦", "贾宝玉", "林黛玉", "第一回"]
    for test_text in test_texts:
        if all(c in stoi for c in test_text):
            encoded = encode(test_text)
            decoded = decode(encoded)
            print(f"  {test_text} -> {encoded} -> {decoded}")
        else:
            missing_chars = [c for c in test_text if c not in stoi]
            print(f"  {test_text} -> 包含未知字符: {missing_chars}")
    
    # 显示一些训练数据样本
    print(f"\n📖 训练数据样本:")
    sample_start = decode(train_data[:100].tolist())
    print(f"前100个字符: {sample_start}")

if __name__ == '__main__':
    prepare_hongloumeng()