# 红楼梦训练配置
# 基于完整的红楼梦文本进行中文语言模型训练

# 数据设置
out_dir = 'out-hongloumeng'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False # 如果为True，脚本会在第一次评估后退出
always_save_checkpoint = True

# 数据
dataset = 'chinese_literature'
gradient_accumulation_steps = 5 * 8 # 用于模拟更大的批次大小
batch_size = 12
block_size = 512 # 红楼梦的句子和段落长度适中，512是个好的选择

# 模型设置 - 针对中文优化
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # 对于从头训练，不使用dropout
bias = False # True: 偏置在LayerNorm和Linear层中. False: 稍好一点且更快

# AdamW优化器
learning_rate = 6e-4 # 对于较小的数据集稍微降低学习率
max_iters = 15000 # 红楼梦数据量不大，不需要训练太久
lr_decay_iters = 15000 # 学习率衰减到max_iters
min_lr = 6e-5 # 学习率的最小值，设为learning_rate/10
beta2 = 0.99 # 对中文效果更好

# 学习率衰减设置
decay_lr = True # 是否在训练过程中衰减学习率
warmup_iters = 1000 # 预热步数，对小数据集来说1000步足够

# 系统设置
device = 'cuda' # 使用GPU训练，若无GPU可用可通过命令行覆盖为 'cpu'
dtype = 'float32' # 使用float32确保稳定性
compile = True # 使用PyTorch 2.0来编译模型以获得更好的性能

# 中文特定设置
init_from = 'scratch' # 'scratch' 或 'resume' 或 'gpt2*'

# 数据加载器设置
num_workers = 0 # DataLoader的工作进程数

# 正则化 - 对于红楼梦这样的经典文本，我们希望模型能够很好地记忆
weight_decay = 1e-1
grad_clip = 1.0 # 梯度裁剪值

# 生成设置（用于测试）
temperature = 0.8 # 生成时的温度参数，0.8对中文比较合适
top_k = 200 # 保留概率最高的k个token

# wandb日志
wandb_log = False # 覆盖via命令行如果需要
wandb_project = 'hongloumeng-nanogpt'
wandb_run_name = 'hongloumeng-' + str(n_layer) + 'L-' + str(n_head) + 'H-' + str(n_embd) + 'D'