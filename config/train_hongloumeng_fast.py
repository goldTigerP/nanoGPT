# 红楼梦快速训练配置 - CPU优化版本
# 针对CPU训练优化，减小模型规模，加快训练速度

# 数据设置
out_dir = 'out-hongloumeng-fast'
eval_interval = 50  # 减少评估间隔，更快看到结果
log_interval = 5    # 更频繁的日志
eval_iters = 20     # 减少评估时间
eval_only = False
always_save_checkpoint = True

# 数据
dataset = 'chinese_literature'
gradient_accumulation_steps = 1  # 大幅减少，提高训练速度
batch_size = 8      # 减小批次大小，减少内存使用和计算时间
block_size = 256    # 减少序列长度，加快训练

# 模型设置 - 大幅减小模型以提高训练速度
n_layer = 6         # 从12减少到6层
n_head = 6          # 从12减少到6头
n_embd = 384        # 从768减少到384维
dropout = 0.0
bias = False

# AdamW优化器
learning_rate = 1e-3    # 提高学习率，更快收敛
max_iters = 1000        # 大幅减少训练步数，从15000到1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99

# 学习率衰减设置
decay_lr = True
warmup_iters = 100      # 减少预热步数

# 系统设置
device = 'cpu'
dtype = 'float32'
compile = False         # 关闭编译，节省启动时间

# 中文特定设置
init_from = 'scratch'

# 数据加载器设置
num_workers = 0

# 正则化
weight_decay = 1e-2     # 减少正则化
grad_clip = 1.0

# 生成设置
temperature = 0.8
top_k = 200

# wandb日志
wandb_log = False
wandb_project = 'hongloumeng-nanogpt-fast'
wandb_run_name = 'hongloumeng-fast-' + str(n_layer) + 'L-' + str(n_head) + 'H-' + str(n_embd) + 'D'