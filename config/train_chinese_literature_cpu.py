"""
中文文学作品CPU训练配置文件
适用于MacBook或其他只有CPU的设备
"""

# 数据集配置
dataset = 'chinese_literature'
gradient_accumulation_steps = 1
batch_size = 12  # 减小batch size
block_size = 64  # 减小序列长度

# 模型配置（较小的模型）
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# 训练配置
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4

# 评估配置
eval_interval = 100
eval_iters = 20
log_interval = 10

# 保存配置
always_save_checkpoint = True

# 系统配置
device = 'cpu'
dtype = 'float32'
compile = False