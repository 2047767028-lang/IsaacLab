# IKEA-Manual 深度分析（2026-09-04）

> 论文：**IKEA-Manual: Seeing Shape Assembly Step by Step**，[arXiv 2302.01881](https://arxiv.org/abs/2302.01881)
> **NeurIPS 2022 Datasets and Benchmarks Track**
> 作者：Ruocheng Wang, Yunzhi Zhang, Jiayuan Mao, Ran Zhang, Chin-Yi Cheng, **Jiajun Wu**（Stanford + MIT + Autodesk）
> 项目页 <https://cs.stanford.edu/~rcwang/projects/ikea_manual/>
>
> 本文所有数字均从论文 PDF 原文（`pdftotext -layout`）与本地解包的数据集中直接提取。

## 1. 他们交付了什么

**数据集**（已完整下载 173 MB 核实）：
- **102 个 IKEA 物体**：Chair 57 / Table 19 / Misc 11 / Bench 8 / Desk 4 / Shelf 3
- **754 个 `.obj` 零件文件**，零件数 min 2 / max 19 / **平均 7.4**
- 每个物体：`connection_relation`（零件两两连接）、`assembly_tree`（装配树）、
  `geometric_equivalence_relation`（几何等价零件）、`steps[]`（每步零件/连接/页码/mask/内外参）
- 附 `pdfs/`（221 页说明书）、`line_seg/`、`mask/`、`code/`（实验代码）
- ⚠️ **zip 内无任何 LICENSE 文件**

**四个任务的基线实验** —— 见下节。

## 2. 他们得到的"结果"：本质上是四个负面结果

> 这是理解这篇论文的关键。它的贡献方式是**"证明现有方法在真实零件上全都不行"**，
> 而不是"我们提出的方法最好"。NeurIPS D&B track 接收了它 —— 说明数据集论文的
> 成功标准是**揭示现有方法的失败**。

### 2.1 装配计划生成（预测装配树）
| 方法 | Simple P | Simple R | Simple F1 | **Hard P** | **Hard R** | **Hard F1** |
|---|---|---|---|---|---|---|
| SingleStep | 100 | 35.77 | 48.64 | 10.78 | 10.78 | **10.78** |
| GeoCluster | 44.90 | 48.46 | 43.53 | 16.54 | 16.50 | **16.30** |

论文原话：**"Neither of the two baselines perform well on the task, which demonstrates its difficulty."**
（SingleStep 的 Simple 精度 100 是伪的——它只输出根节点，而根节点必然存在。）
GeoCluster 只看几何相似度，**无法考虑零件间的连接约束**。

### 2.2 说明书零件分割（Part-Conditioned Manual Segmentation）
353 训练 / 40 测试，微调 Li et al.（ECCV 2020，U-Net + PointNet，在 PartNet 椅子上预训练）。
- **IoU = 25.31**
- 论文："shows that Li et al. struggles on the task"，对**由多个基本零件组合而成的部件**尤其吃力。

### 2.3 零件位姿估计（Part-Conditioned Pose Estimation）
1056 个样本（844 训 / 105 验 / 107 测）。

| 方法 | 旋转误差↓ | 旋转准确率↑ | CD↓ | CA↑ | 推理时间↓ |
|---|---|---|---|---|---|
| Random | 103.49° | 3.05% | 0.1344 | 0.28 | — |
| Xiao et al. | 82.79° | 20.56% | 0.0984 | 49.53 | < 1e-5 s |
| SoftRas | **64.81°** | **24.30%** | 0.0976 | 51.4 | **243 s** |

都显著高于随机，但论文："there is plenty of room for improvement"。
SoftRas 略好但**慢 7 个数量级**（推理时做多次优化）。

### 2.4 零件装配（Part Assembly）—— **最有价值的一个结果**

| 方法 | PartNet Chair CD↓ | PartNet **零件准确率↑** | IKEA-Manual Chair CD↓ | IKEA-Manual **零件准确率↑** |
|---|---|---|---|---|
| B-LSTM | 0.0131 | 21.77 | 0.0181 | **3.48** |
| B-Global | 0.0146 | 15.70 | 0.0195 | **0.87** |
| RGL-Net | N/A | N/A | 0.0583 | **2.01** |
| RGL-Net\* | 0.0087 | **49.06** | 0.0508 | **3.99** |
| Huang et al. | 0.0091 | **39.96** | 0.0151 | **6.90** |

**零件准确率从 PartNet 的 40~49% 崩到 IKEA-Manual 的 3~7%。**

论文的解释：PartNet 的零件是 **"simplified versions of the real-world parts and have less variety"**；
模型能把零件拼成"像椅子的整体形状"（所以 CD 只轻微变差），
**但认不出每个零件的语义角色**（所以零件准确率崩塌）。

> 👉 **这个 40% → 4% 的崩塌是整篇论文对我们最有用的数字**：
> 它定量证明了"简化零件 vs 真实零件"的巨大差距，
> 正好是我们"真实尺度资产"这个卖点的实证支撑。

## 3. 论文自己承认的局限 —— 这是留给后人的口子

Discussion 原文两条：

1. **"One limitation of our work is that we do not model connectors between assembly parts."**
   —— **他们不建模连接件**。并明说 "Incorporating connector modeling can enable shape assembly
   tasks that align better with real-world scenarios."
2. **数据严重偏向 Chair**（57/102），"reflects the imbalanced distribution of online 3D repositories"。

## 4. 对我们最关键的判断：IKEA-Manual 里没有机器人、没有物理、没有仿真

四个任务全是**感知与规划**：装配树预测、图像分割、位姿估计、点云装配。
**没有任何一个任务是"机器人真的把它装起来"。**

所以它提供的是 **"图纸 + 零件几何"**，不是 **"能装配的物理资产"**。
从 IKEA-Manual 到 Isaac Lab 里能跑的东西，中间隔着四道工序：

| 缺什么 | 说明 |
|---|---|
| **配合特征** | 没有榫卯 / 销孔 / 螺纹 —— 论文明说不建模连接件。**这是最难补的一道** |
| **物理属性** | 没有碰撞体、质量、摩擦、惯量 |
| **仿真格式** | 只有 `.obj`，没有 URDF / USD |
| **真实尺度** | OBJ 是任意尺度，需要对应实物尺寸重新标定 |

## 5. 三年后的影响：35 次引用，5 次高影响力引用

后续谱系里跟我们相关的几条：

| 工作 | 年份/会议 | 引用 | 关系 |
|---|---|---|---|
| **Manual2Skill** | RSS 2025 | 22 | **从 IKEA-Manual 走向机器人的主干线** |
| **Manual2Skill++** | 2025-10（2026-03 修订） | 4 | **connector-aware**，正好补上"连接件"这个口子；做了 Isaac Lab 上的装配 benchmark |
| **AssemblyBench** [2605.12845](https://arxiv.org/abs/2605.12845) | **CVPR 2026**（MERL + FAU） | 1 | **2,789 个工业物体 + 多模态说明书 + CAD 零件 + 物理可行的 6-DoF 装配轨迹**，用**物理仿真**评估轨迹可行性。提出 AssemblyDyno 模型。**已开源**（`merlresearch/AssemblyBench`）。这是把 IKEA-Manual 缺的**物理维度**补上的那篇 |
| IKEA Manuals at Work | NeurIPS 2024 | 15 | 同组后续，说明书 4D grounding 到网络视频 |
| Manual-PA | ICCV 2025 | 11 | 说明书图引导的 3D 零件装配 |
| SPAFormer | 2024 | 11 | Transformer 做顺序零件装配 |
| Two by Two | CVPR 2025 | 20 | 多任务成对物体装配 |
| BiAssemble | ICLR 2025 | 5 | 双臂几何装配 |

**趋势很清楚**：IKEA-Manual（2022，纯视觉/几何）→ Manual2Skill（2025，接机器人）
→ Manual2Skill++（连接件感知）→ AssemblyBench（2026，物理轨迹 + 仿真评估）。
**整条线在往"物理可执行"的方向收敛**，而这正是我们要落脚的地方。

## 6. 结论：怎么用它

**能用**：102 件真实家具的零件分解 + 连接关系 + 装配树 —— 这是**任务定义层**的宝贵输入
（哪些零件、按什么顺序、连到哪里），省掉大量人工标注。

**不能直接用**：它的 `.obj` 不是能在物理引擎里装配的资产。要用必须补配合特征、物理属性、
USD 转换和尺度标定 —— **这四道工序本身就是我们 benchmark 的资产工作量所在**。

**阻塞**：**许可未知**（zip 内无 LICENSE）。必须联系作者 Ruocheng Wang（Stanford）确认。
