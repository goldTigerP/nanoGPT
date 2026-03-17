# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

---

**2025年11月更新** nanoGPT有了一个新的改进版本叫做[nanochat](https://github.com/karpathy/nanochat)。很可能你想要使用/寻找的是nanochat。nanoGPT（这个仓库）现在已经很老旧且已弃用，但我会保留它用于纪念。

---

训练/微调中等规模GPT最简单、最快速的代码仓库。这是[minGPT](https://github.com/karpathy/minGPT)的重写版本，优先考虑实用性而非教育性。目前仍在积极开发中，但当前的`train.py`文件可以在单个8XA100 40GB节点上用大约4天时间在OpenWebText上重现GPT-2（124M）。代码本身简洁易读：`train.py`是一个约300行的样板训练循环，`model.py`是约300行的GPT模型定义，可以选择性地加载来自OpenAI的GPT-2权重。就这么简单。

![repro124m](assets/gpt2_124M_loss.png)

由于代码非常简单，很容易根据你的需求进行修改，从头训练新模型，或者微调预训练检查点（例如，目前可用作起点的最大模型是来自OpenAI的GPT-2 1.3B模型）。

## 安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖包：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` 用于huggingface transformers <3 (加载GPT-2检查点)
-  `datasets` 用于huggingface datasets <3 (如果你想下载+预处理OpenWebText)
-  `tiktoken` 用于OpenAI的快速BPE编码 <3
-  `wandb` 用于可选的日志记录 <3
-  `tqdm` 用于进度条 <3

## 快速开始

### 中文文学作品训练（推荐中文用户）

如果你是中文用户，可以在红楼梦、水浒传等中文古典文学作品上训练GPT模型。首先准备中文数据：

```sh
cd data/chinese_literature
python prepare.py
```

这会创建基于中文字符的`train.bin`和`val.bin`文件。然后开始训练：

**GPU训练:**
```sh
python train.py config/train_chinese_literature.py
```

**CPU训练（MacBook等）:**
```sh
python train.py config/train_chinese_literature_cpu.py
```

**使用预训练模型微调（推荐）:**
```sh
python train.py config/finetune_chinese_literature.py
```

训练完成后，生成中文文本：
```sh
python generate_chinese.py --start="红楼梦第一回" --num_samples=3
```

### 英文莎士比亚作品训练

如果你想使用原始的莎士比亚作品训练，最快的入门方法是训练一个字符级GPT。首先，我们将其下载为单个（1MB）文件，并将其从原始文本转换为一个大的整数流：

```sh
python data/shakespeare_char/prepare.py
```

这会在该数据目录中创建`train.bin`和`val.bin`文件。现在是时候训练你的GPT了。模型的大小很大程度上取决于你系统的计算资源：

**我有GPU**。很好，我们可以使用[config/train_shakespeare_char.py](config/train_shakespeare_char.py)配置文件中提供的设置快速训练一个小型GPT：

```sh
python train.py config/train_shakespeare_char.py
```

如果你查看配置文件内部，你会看到我们正在训练一个上下文大小为256个字符、384个特征通道的GPT，它是一个6层Transformer，每层有6个注意力头。在一块A100 GPU上，这个训练运行大约需要3分钟，最佳验证损失是1.4697。根据配置，模型检查点被写入`--out_dir`目录`out-shakespeare-char`。所以一旦训练完成，我们可以通过将采样脚本指向这个目录来从最佳模型中采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

这会生成一些样本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

哈哈 `¯\_(ツ)_/¯`。对于一个在GPU上训练3分钟后的字符级模型来说还不错。通过在这个数据集上微调预训练的GPT-2模型很可能获得更好的结果（参见后面的微调部分）。

**我只有一台MacBook**（或其他便宜的电脑）。别担心，我们仍然可以训练GPT，但我们需要调低一档。我建议使用最新的PyTorch nightly版本（[在这里选择](https://pytorch.org/get-started/locally/)安装），因为它目前很可能让你的代码更高效。但即使没有它，一个简单的训练运行可能如下所示：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

这会在你的MacBook上运行，大约需要90分钟。由于上下文长度较短（64），模型很小，验证损失只能达到1.88，但这仍然意味着模型学到了英语的拼写和一些词汇结构。给定更多的时间训练（如果你将max_iters设置得更高），你可能会得到更好的结果。

你还可以通过以下命令在更小的数据集上进行实验：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=5 --log_interval=10 --block_size=32 --batch_size=1 --max_iters=100 --lr_decay_iters=100 --dropout=0.0
```

## 中文文学作品训练详解

### 数据准备

中文训练使用字符级tokenization，这对中文文本效果很好：

1. **自动下载**: 脚本会尝试下载红楼梦、水浒传、三国演义、西游记
2. **本地文本**: 你也可以将自己的中文文本文件放在`data/chinese_literature/`目录下
3. **字符编码**: 支持所有中文字符、标点符号和换行符

### 训练配置选择

**GPU用户（推荐）:**
- 模型：6层Transformer，384维嵌入
- 训练时间：约30-60分钟（取决于GPU性能）
- 内存需求：4-8GB显存

**CPU用户:**
- 模型：4层Transformer，128维嵌入  
- 训练时间：2-4小时
- 内存需求：8GB系统内存

**微调模式（最佳效果）:**
- 基于GPT-2预训练模型
- 训练时间：10-30分钟
- 生成质量更好

### 生成示例

训练完成后，你可以生成类似这样的中文文本：

```
红楼梦第一回：
话说女娲氏炼石补天之时，于大荒山无稽崖练成高经十二丈、方经二十四丈顽石三万六千五百零一块。
那娲皇只用了三万六千五百块，单单剩了一块未用，便弃在青埂峰下...
```

## 在OpenWebText上训练

如果你想在类似GPT-2的数据集上训练，你可以下载并使用OpenWebText数据集，它是WebText的开源复制品（GPT-2训练的数据）：

```sh
python data/openwebtext/prepare.py
```

这将下载并tokenize OpenWebText数据集。根据你的网络连接，这可能需要一些时间（数据集约8GB）。默认情况下，它会下载到`data/openwebtext/`。

准备好数据后，我们可以开始训练。为了重现GPT-2（124M），你将需要至少一个8XA100 40GB节点，训练时间约为4天。配置在[config/train_gpt2.py](config/train_gpt2.py)中：

```sh
python train.py config/train_gpt2.py
```

## 采样/推理

给定任何模型，例如从莎士比亚微调得到的模型，你可以使用以下命令生成无限量的样本：

```sh
python sample.py \
    --init_from=gpt2-medium \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

## 配置

配置系统有点独特，但非常简单。我们不使用配置文件或YAML文件，而是使用Python文件。这样你就可以在配置中进行任何计算、导入或其他任何操作。看看`config/`目录中的示例。

配置文件只是覆盖`train.py`中的默认参数。例如，如果你想将学习率设置为3e-4，你可以创建一个包含以下内容的配置文件：

```python
learning_rate = 3e-4
```

然后运行：

```sh
python train.py your_config.py
```

你也可以通过命令行覆盖参数，这将覆盖配置文件中的设置：

```sh
python train.py your_config.py --batch_size=32
```

## 基准测试

nanoGPT的性能与其他实现相当。训练GPT-2（124M）在OpenWebText上的速度：

- nanoGPT: ~4天在8xA100 40GB上
- GPT-2原始: ~4天在8xV100 32GB上（根据论文）

## 故障排除

一些常见问题及其解决方案：

**RuntimeError: CUDA内存不足**
- 减少batch_size或者使用gradient accumulation
- 减少模型大小（n_layer, n_embd等）

**ModuleNotFoundError: No module named 'transformers'**
- 安装依赖：`pip install transformers`

**训练非常慢**
- 确保你使用的是GPU：`--device=cuda`
- 启用编译：`--compile=True`（需要PyTorch 2.0+）

## 许可证

本项目采用MIT许可证。

## 鸣谢

本项目深受以下工作启发：
- OpenAI的GPT-2论文和代码
- Hugging Face的transformers库
- 以及深度学习社区的无数贡献

## 贡献

欢迎贡献！请随时提交issue或pull request。