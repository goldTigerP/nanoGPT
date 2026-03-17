"""
中文文学作品微调配置文件
基于预训练GPT-2模型进行中文文学作品微调
"""

# 初始化配置
init_from = 'gpt2' # 'gpt2', 'gpt2-medium', 'gpt2-large' or 'gpt2-xl'

# 数据集配置
dataset = 'chinese_literature'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# 微调特定配置
learning_rate = 5e-5 # 微调时使用较小的学习率
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-6

# 评估配置
eval_interval = 100
eval_iters = 100
log_interval = 10

# 保存配置
always_save_checkpoint = True
out_dir = 'out-chinese-literature-finetune'

# 系统配置
device = 'cuda' # 如果只有CPU则改为'cpu'
dtype = 'bfloat16'
compile = True