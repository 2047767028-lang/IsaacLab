# 主线一 benchmark 定位调研（2026-09-03，v2）

> v1 的核心推荐（"把扰动敏感性/超越单点成功率当卖点"）在拿到学长的调研清单后**已作废**，
> 原因见 §3。本版是重写。v1 保留在 git 历史里（commit 5921d70e3）。
> arXiv 条目均以 `curl https://arxiv.org/abs/<id>` 抓原文摘要核对，不用搜索引擎转述。

## 1. 学长文档里的计划（用户 2026-09-03 提供正文）

未来工作两条：
1. **G2/G1 真机**：强化学习微调；优化售货系统、交互体验、泛化抓取；世界模型部署。
2. **家具装配 benchmark**：
   - 框架基座：场景、机器人 URDF、task、评测标准
   - 数据集：遥操作采集、强化学习采集、mimic 数据扩充
   - **3D 资产制作：可拆分的 3D 家具资产**（Blender、codex、淘宝、**团队工作**）

学长另做了 10 个"基于 Isaac Sim 的 benchmark"调研，外加一张
**"benchmark 基础框架优劣"对比图（图片，未拿到内容——需要问学长选型结论）**。

## 2. 竞品格局（学长清单 + 我补充核实的）

### 2a. 通用操作 benchmark：极度拥挤，不要进
| 工作 | 时间 | 关键点 |
|---|---|---|
| **RoboLab** [NVLabs](https://github.com/NVLabs/RoboLab) | **RSS 2026-07** | NVIDIA 自己做的。建在 Isaac Lab 上，**RoboLab-120** 共 120 任务（pick-and-place/堆叠/重排/工具使用），视觉·程序·关系三维度 × 三难度。**明确量化"策略行为对受控扰动的敏感性"**，开场就在批"现有 benchmark 训练/评估域重叠、成功率虚高、掩盖鲁棒性洞察"。机器人与策略无关。**不含装配。** |
| **REALM** | 2025-12 | 15 个扰动因子、7 种技能、3500+ 物体，评测 π0 / π0-FAST / GR00T N1.5，"系统性探查并量化 VLA 的弱点与失效模式"。 |
| **RoboDojo** | 2026 | 42 仿真 + 18 真机任务，5 个能力维度，Isaac Sim 异构并行，RoboDojo-RealEval 云端真机复现，XPolicyLab 统一策略接入，**已接 30 个策略做公开排行榜**。 |
| **EBench**（InternRobotics） | 2026 | 26 任务，5 能力维 + 4 泛化维，**"评估维度不仅限于单一成功率标量"**，诊断 π0/π0.5/XVLA/InternVLA-A1。 |
| **RoboTwin 2.0**（IsaacLab-Arena 分支） | 2025-06 | 双臂，RoboTwin-OD 147 类 731 实例，5 维结构化域随机化，50 个双臂任务 × 5 种本体，数据/基准/代码全开源。 |
| **LW-BenchHub**（Lightwheel） | 2026 | 建在 **Isaac Lab-Arena** 上的统一基准中心。 |
| **UniVTAC** | 2026 | 视触觉，8 个任务，含插入类接触密集操作。 |

### 2b. 装配 benchmark：反而是空的
| 工作 | 状态 |
|---|---|
| **FurnitureBench** [RSS 2023](https://arxiv.org/abs/2305.12821) | 唯一的家具装配基准。但：**Isaac Gym 时代**（已弃）、**单臂 Franka**、**3D 打印的玩具尺度**零件、只有 3 个任务（one-leg table 4 步 / round table 8 步 / lamp 11 步）。**开放 3D 资产**——这是可以直接用的部分。 |
| **Manual2Skill++** [2510.16344](https://arxiv.org/abs/2510.16344) | Isaac Lab 上 4 个装配任务（IKEA 椅 6 件 22 步 / 鞋架 / 飞机模型 / 乐高），含榫卯、木销、螺丝。但它是**方法论文**（VLM 读说明书），benchmark 是载体，不是维护中的发布。 |
| **A3D** [2601.11076](https://arxiv.org/abs/2601.11076) | 双臂家具装配，8 类家具 50 零件的仿真环境。同样是**为自己方法自建**的环境。 |
| **FurnitureVLA** [2607.01212](https://arxiv.org/abs/2607.01212) | 2026-07。真实尺度双臂家具装配 + 仿真数据生成流水线 + 单操作员双臂 VR 遥操作。7 子任务 / 1550 控制步。**未找到开源代码或项目页**，且是方法论文（VLA）不是基准/资产发布。 |
| **Forge / AutoMate** | 已在 Isaac Lab 仓库内，但是**工业件**（插孔、齿轮、螺母），不是家具。 |

**关键观察：2a 那一整波 2025–2026 的通用 benchmark，没有一个碰家具装配。**
不是因为不重要，是因为**装配需要可拆分、有真实连接件、物理上装得进去的 3D 资产——那是人力活，不是算法活。**
这正好是团队有、单人研究者没有的东西（学长的计划里已经写了"团队工作"）。

## 3. v1 推荐为什么作废

v1 建议把"评测协议的噪声/超越单点成功率/扰动敏感性"当论文卖点。**这条已被三方占据**：
- **RoboLab**（NVIDIA 自家，RSS 2026）直接做"量化策略对受控扰动的敏感性"；
- **REALM** 做 15 因子扰动 + 失效模式量化；
- **EBench** 明说"不止单一成功率标量"；
- 而且 **Isaac Lab Arena 本身就内建 sensitivity analysis**。

结论：**这已经是 table stakes，不是 novelty。不能当卖点写。**

## 4. Isaac Lab Arena：框架基座这块不要自己造

- 官方定位：可组合、可扩展的仿真与**策略评测**框架，"NVIDIA 与合作伙伴都在统一的 Arena core 上建基准，
  **task / scene / metric / dataset 都是可复用积木**"。
- 已提供：task API（composite & sequential task、**predicates**、**subtask progress**）、metrics、
  **IsaacTeleop 遥操作**、**data generation（含遥操作数据采集）**、**并行评测（比串行快 13.5×，
  复杂任务从 10 小时降到 1 小时以内）**、多节点评测、sensitivity analysis。
- 已迁移上去的：RoboCasa Tasks、RoboFinals、LW-BenchHub、RoboTwin（IsaacLab-Arena 分支）。
- **Arena 目前没有装配 / 家具任务。**

→ 学长计划里的"框架基座：场景、机器人 URDF、task、评测标准"**绝大部分是 Arena 的现成积木**。
→ 而且 Arena 的遥操作走 **IsaacTeleop**，正是我们服务器上已经打通的那一套（见 `RUNBOOK_mimic.md`）。

## 5. 建议的定位（对学长计划的三条修正）

**总体：学长的计划方向是对的——空地确实在装配，而且门槛正好是团队能跨、个人跨不过的资产人力。**
需要修正的是三点：

1. **不要自己造框架基座，建在 Isaac Lab Arena 上。**
   省下的时间全部投到资产和数据——那才是别人抄不走的。附带好处：RoboCasa / RoboTwin /
   LW-BenchHub 的用户能零成本跑我们的任务，采用率完全不是一个量级。
   ⚠️ 前提：学长那张"框架基座优劣"对比图的结论是什么？如果他已经选了 Arena，这条就是重复劳动。

2. **论文卖点写"真实尺度 + 可拆分 + 多连接件类型 + 双臂"，不要写"超越成功率/鲁棒性诊断"。**
   靶子明确对准两个：
   - vs **FurnitureBench**：它是玩具尺度、Isaac Gym、单臂、3 个任务；我们是真实尺度、Isaac Lab Arena、双臂、任务数更多；
   - vs **FurnitureVLA**：它是真实尺度双臂但**闭源**，且是方法论文；我们是**开放的资产 + 任务 + 数据集发布**。
   - vs **2a 那一波**：它们根本没有装配。

3. **主线二的成果降级但不要扔——转成数据集发布的质量背书。**
   我们实测过 MimicGen 生成数据里 **7.4% / 5.3% 的"成功 demo"末帧是散的**（成功判据只验瞬时几何、
   不要求静止，方块自由落体途中就判成功），并已提 upstream PR（#7434、#7433，证据在 `docs/mimic_pr_evidence/`）。
   → 可以成为**第一个公布并修正了假成功率的装配数据集**。RoboTwin 那种大规模合成数据发布如果没做这件事，
   这就是可写的差异点。这是 release 的可信度特性，**不是论文主线**。

## 6. 不依赖任何人就能开始的下一步

1. 装 Isaac Lab Arena，把服务器上已打通的 IsaacTeleop 链路接到 Arena 环境里跑通一次。
2. 在 Arena 上做出 **one-leg table 双臂版**（FurnitureBench 最简那个，4 步）作为最小闭环，
   验证"遥操作采集 → mimic 扩充 → 并行评测"整条流水线。
3. 资产：先确认 **FurnitureBench 的 3D 模型许可**能不能直接用/改——能用就省掉第一批建模人力。

## 7. 待确认

1. **学长那张"benchmark 基础框架优劣"对比图的结论**——他选型选了什么？（决定 §5.1 是不是重复劳动）
2. **资产制作的人力规模**——决定任务数量目标（FurnitureBench 3 个 / Manual2Skill++ 4 个 / A3D 8 类 50 件）。
3. **目标会议与时间线。**
4. 服务器 `surf2@192.168.3.254` 目前 SSH 连不上（`kex_exchange_identification: Connection closed`，
   试了 3 次），恢复后需要核实服务器上主线一的实际进度。
