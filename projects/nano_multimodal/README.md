# nano_multimodal — 三个模态,一套自回归训练

三个模型:**文本语言模型**、**文本→动作**、**视频世界模型**,外加文本线上的
**compute-optimal scaling law**(`scaling.py`,5 档模型 × 500M token,两张卡 38 分钟)。
tokenizer 和编码好的数据都是给定的,所以三条线在训练侧是同一件事——整数序列进、
整数序列出的自回归模型。两个 web 面板让这件事**看得见**:训练时喂进去的是什么,
训出来的又是什么。

要读代码,从 `spec.py` → `train.py` → `configs/` 三份 yaml 开始。

## 这三个模型跑出来是什么样

单张 RTX 5090,三条线各一次完整训练:

| 线 | 模型 | 步数 | 训练时长 | 监督 token | 最好 val CE |
|---|---|---|---|---|---|
| 文本(先跑通) | d6 / 35.8M | 3,000 | ~15 分钟 | 393M | 4.1561 |
| 文本(冠军) | d12 / 135.3M | 20,640 | 4.6 小时 | 2.705B | **3.4113** |
| 动作 | d6 / 36.2M | 16,000 | 8.7 分钟 | 131M | 2.3685 |
| 视频 | d12 / 183.3M | 9,059 | 62 分钟 | 377M | 5.2476(起点 11.07) |

参数量是**整个系统**(trunk + LM head),也就是训练器启动时打印的那个数。同样是
d12/768,三条线的参数量不同 —— 差的全在 un-embedding 上,词表越宽它越大
(文本 32,768 / 动作 33,280 / 视频 64,037)。只数 trunk 会得到另一套数(d12/768
都是 110M),两套数不要混进同一张表。

文本冠军的步数不是手填的:`max_steps: -1` 让 core 按参数量算 Chinchilla 预算
(135M × 20 = 2.705B token),再除以每步 token 数。它的最好 val **就是最后一步** ——
喂满 Chinchilla 预算都还没到转折点。

★ **动作线上把模型放大不买东西,这是这门课的中心一课。** 同样的语料同样的配方,
d12(136.1M,是 d6 的 3.8 倍)在第 5,500 步就到 val 2.3811,然后一路漂到 22,000 步的 2.6943;而 d6
到 16,000 步还在降(2.384 → 2.3685),**始终没被 d12 超过**。
原因可量:Bones-SEED 一个 epoch 是 450,594 行 = 30.4M 监督动作 token,
而一个 36.2M 的模型按 Chinchilla 想要 ~720M(约 24 遍)。**语料是上限,不是模型。**
推理面板的 checkpoint 选择器把 d12 的「谷底」和「末尾」都标了出来,切着看就是这一课。

★ **同样的 12 层 / 768 宽,在文本上不过拟合、在动作上过拟合** —— 上表两行放在一起
就是证据。
所以"过拟合"不是模型的属性,是**模型容量与语料量之比**的属性。

数据体积:视频 token 子集 737 MB、动作 cache 113 MB、文本 tokenizer 0.7 MB;
解码用的 Cosmos decoder 120 MB、动作 codec 70 MB、SMPL 静止骨架 1.9 KB。

### 文本线的 scaling law

五档模型(depth 2/3/4/6/8,dim = 64×depth)各喂 **500M token**,恒定学习率,
两张 5090 跑 38 分钟。前沿指数 **a = 0.446**(N_opt ∝ C^a),包络上有 4 处曲线交叉。

| depth | 非嵌入参数 | 最终 val CE |
|---|---|---|
| 2 | 0.4M | 5.1678 |
| 3 | 1.3M | 4.8097 |
| 4 | 3.1M | 4.6152 |
| 6 | 10.6M | 4.3726 |
| 8 | 25.2M | 4.2065 |

★ **这个数字不能脱离预算引用。** 前沿指数随覆盖的算力增长,因为包络的高算力端还在填:
把同一批曲线截断后重拟,100M 处是 0.566、300M 是 0.468、500M 是 0.446;
`exemplars/text_pretrain` 的 2B 研究在 1B 处 0.507、2B 处 0.519。
**a ≈ 0.45 @ 500M 不等于"Chinchilla 0.5 已复现"**——`scaling.py fit` 因此总是把预算和
数字一起打印。同口径对照下,本项目在 300M 处与那份参照实现完全一致(两边都是 0.468)。

## 三条线的全部差别

```
                  ①激活带              ②数据源            ③一行怎么摆        ④哪些位置算 loss
文本    text, control              TextDataSource     bos text… eos          全部
动作    text, control, motion      T2MDataSource      caption + 码           动作半边
视频    control, video, action     VideoRowSource     帧码与动作交错          被预测的 4 帧

以下三条线完全相同:1D RoPE · next-token CE · GPT+LMHead · MixedDataLoader · core Trainer
```

`train.py` 里没有一处 `if line == ...`。把三份 config diff 一下就看得见为什么。

## 数据从哪来

三条线的 tokenizer 和编码好的数据都是**给定的**——这正是让三条线在训练侧完全相同的
前提。但"给定"要说清是谁给:

| 线 | 要什么 | 怎么拿到 |
|---|---|---|
| 文本 | FineWeb parquet | 公开。放进 `outputs/base_data/`。d6 教学跑 3,000 步只吃 0.39B token,**一个 shard 就够**;复现 d12 冠军点(2.705B)要 4 个 |
| 动作 | 预编码的 `outputs/motion_caches/t2m_bones_*.npz` + `models/motion/codec_rot139_kin_fsq2_*.pt` | **上课发的数据包里有**。要自己造:走 `exemplars/nano_motion` 的 `data/encode.py`(它记着自己那条语料的取法) |
| 视频 | `outputs/cache/dv128_17f/` 的 memmap + 解码用的 Cosmos `decoder.jit` | **数据包里有**。要自己造:`exemplars/nano_world_model/data/download.py` 抓公开的 VizDoom 数据集和 codec,再 `encode.py` → `build_cache.py`,得到的 cache 直接用 `video_cache=` 覆盖(见 `configs/video.yaml`) |

本目录的 `data/build_video_cache.py` 从 `datasets/pipe4` 切子集——那是**内部的主线语料**,
公开读者没有。它留在这里是为了说明子集是怎么切的(以及为什么切成"一小时正好一遍"),
不是给外部复现用的入口;外部入口是上表视频那一行的 exemplar 链路。

渲染火柴人还要 `models/smplh/neutral/model.npz`,但只用到里面的静止骨架和父节点表
(**1.9 KB**,数据包里带的就是这个裁剪版)。完整 SMPL+H 要去
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) 自己取,本项目三条线都不需要它。

## 跑

```bash
# 0. 文本 tokenizer(几秒),它给整个共享词表定尺寸
#    数据包里已经带了一份;要自己训就跑这行(vocab 必须仍是 32768)
python -m modalities.text.train_tokenizer

# 1. 训练:三条线,同一个编排器
python -m projects.nano_multimodal.train --config-name motion   # 分钟级
python -m projects.nano_multimodal.train --config-name video    # ~1 小时(单 A100)
python -m projects.nano_multimodal.train --config-name text     # ~4 小时

# 两张卡:launcher 决定卡数(RANK 只有 torchrun 会设),parallel 决定摆法
torchrun --nproc_per_node=2 --standalone \
    -m projects.nano_multimodal.train --config-name video parallel=ddp

# 2. 文本的 compute-optimal scaling law(ladder 天然可拆,一张卡一档)
python -m projects.nano_multimodal.scaling run --depths 8 6
python -m projects.nano_multimodal.scaling fit

# 3. 数据浏览器 + 推理 web(同一个进程、同一个端口)
python -m projects.nano_multimodal.serve.app --port 8800 --device cuda:1
```

`--device` 默认指向第二张卡:web 常驻三个模型加三个解码器,和训练挤同一张卡会 OOM,
而且报错不会告诉你原因。

## 状态

**六步全部完成,端到端可跑。** 起始 loss 三条线全部等于 `ln(vocab)`;视频行与
`exemplars/nano_world_model` 的实现逐位一致(`tests/test_video_row.py`);2 卡 DDP 与
续跑均已验证。三个解码器目检通过;三档采样器 greedy 逐 token 一致,CUDA graphs 3.5×
(0.25 s/帧,交互 rollout 4 fps);数据浏览器和推理 web 三条线都跑通,
`/api/row` 与训练 loader 逐 id 相等。

**还没做的,说清楚**:本项目的 scaling 阶梯与 `exemplars/text_pretrain` 的曲线之间有
一处未解释的系统偏差(等预算下本项目的 val 低 0.03–0.10 nat,且随模型增大)。已排除
val 划分、分隔符占比、打包方式;最可能是两份曲线不在同一个词表下,未验证。
**在弄清楚之前,别把两条曲线画在同一张图上。**

`projects/` 是有意的落点——`exemplars/` 的门槛是"跑通过、数字钉过、别人照做能复现";
三条都满足了再搬。

项目**写**出来的东西(checkpoint、视频子集、scaling 曲线、目判画廊)全部落在
`outputs/`;项目根目录只放**发**出去的东西。这条规则由 `spec.py` 里的路径执行,不靠人记,
而 `projects/*/outputs/` 正是本仓库 `.gitignore` 已经管住的地方 —— 一个装着几十 GB 权重的
目录,名字必须是 git 早就被告知别碰的那个。

`outputs/` 不进版本库,所以这一份 README 是自包含的:上面那张表就是全部实测数字。
