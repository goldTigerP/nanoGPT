"""
中文文学作品训练配置文件
适用于红楼梦、水浒传等中文古典小说
"""

# 数据集配置
dataset = 'chinese_literature'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # 中文字符序列长度

# 模型配置
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# 训练配置
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4

# 评估配置
eval_interval = 250
eval_iters = 200
log_interval = 10

# 保存配置
always_save_checkpoint = True

# 系统配置
device = 'cuda' # 使用GPU，如果只有CPU则改为'cpu'
dtype = 'bfloat16'
compile = True # 需要 PyTorch 2.0+