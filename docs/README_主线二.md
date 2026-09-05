# 主线二：从哪读起

六份文档各有分工，重叠不多但入口不明显。按你要做的事挑：

| 你想干什么 | 读哪份 |
|---|---|
| **先看两天的结论** | `主线二_接触帧驻留_结论汇总_2026-09-02至04.md` — 自包含：机制、数字、撤回、原则、下一步 |
| **接着往下推理** | `主线二_思维链交接.md` — 信念怎么变的、哪些确信哪些悬着、下一步的逻辑分叉 |
| 查某个数字/事实 | `CLAUDE.md` 的主线二节 — 事实与数字快照 |
| 知道 arc 为什么伤成功率 | `arc_sweep_diagnosis/README.md` — 机制诊断的完整论证 |
| 知道哪些修法试过、为什么都失败 | `arc_sweep_diagnosis/REMEDIES.md` |
| **引用任何一个实验数字之前** | `arc_sweep_diagnosis/EXPERIMENT_LEDGER.md` — **必读**，见下 |
| 理解最初的设计意图 | `接触锚定扰动增广_设计记录.md` + 末尾的订正节 |
| 上游 PR 的证据与复核 | `mimic_pr_evidence/EVIDENCE.md` |

## 三个"别踩"的提示

**① 引用实验数字前先看 ledger 的分组。** 同一个名义配置（arc 3.0cm、无干预）在不同 harness
设置下给出 28.0% / 22.0% / 15.7%——**12.3pp 的差距全来自并行环境数和是否重新播种**，比大多数
干预的真实效应还大。跨组比较会直接得出错误结论。

**② 设计文档的核心表述已被推翻，但原文保留着。** "接触段完全冻结"只对**目标位姿**成立，对
**机械臂实际状态**不成立（约 45% 的扰动幅度穿过冻结段）。原文没删是因为后面整套设计都建立在
那个假设上，删了会看不懂。**读设计文档必须连订正节一起读。**

**③ CLAUDE.md 里有几条已作废的旧结论**，都用 ⚠️ 标了，尤其"11% 夹爪判据误杀、放宽可白赚
+5pp"那条——**是错的**，照做会把摔方块的 episode 当成功演示写进训练数据。

## 当前状态（2026-09-02）

- 第二层验证结论：扰动增广既无害也无益，效应 −1.83pp、95%CI [−7.11, +3.45]
- 生成成功率下降的机制已查清：**冻结的是指令不是状态**，且 MimicGen 重放的是源演示的**目标
  位姿**而非实际位姿，那约 5cm 的跟踪滞后是重放机制的一部分、不是误差
- ~~五个修法全部失败；损伤的载体三个候选全被排除，未知~~（已作废，见下一条）
- 两个 upstream bug 已提 PR（#7433 / #7434），另有 #7381 已合并、#7380 在审
- **2026-09-03 更新**：载体找到了——接触帧的水平位置。在每次夹爪指令前插 20 帧无噪声驻留（停在
  名义目标上，不需要平行组），arc 3.0cm 生成成功率 15.7% → 39.0%，0.5cm 参照 32.3% → 50.3%。
  上面"载体未知""五个修法全部失败"两条据此**作废**，详见台账 Group E 和 CLAUDE.md 2.14。
  剩余差距来自接近段：冻结比例 30% → 50% 后 3.0cm 与不加 arc 持平（49.7%）；门控驻留再 +5pp
  （55.0%）；1.2cm 生产工作点驻留后 50.0%。推荐组合与未跑项见 CLAUDE.md 2.15、台账 Group F。
- **2026-09-04 晚定案**（CLAUDE.md 2.17）：损失出在抓取段的接近段（闭合时方块已偏在指间），搬运段承受得住
  扰动；扰动保留全轨迹，鼓包返回要在冻结段前几个时间常数结束（峰值提前 + 冻结段随幅度走），验收指标
  = 闭合时偏心回到 0.61cm。术语：抓取段/搬运段/冻结段/接近段。
- **2026-09-04 深夜验证**（CLAUDE.md 2.18，台账 Group I）：峰值提前到 25% 后，5cm 48~51%、10cm 46.7%
  （对照 26% / 6%，无扰动 51.5%），实际弧线不变，闭合偏心 0.68cm。原则成立。
- **2026-09-04 深夜**：10cm 峰值位置扫描单调（0.5/0.35/0.25/0.17 → 6/15/31/38%），返回时间是杠杆；两处待做的
  小代码改动（方向法向化、按实际幅度标定）和第二层验证的测试轴（偏离恢复）见 `docs/mainline2/STATE.md`。


## 怎么在这个工作区继续干活（2026-09-04）

- **在哪**：worktree `/home/pk/IsaacLab/~/IsaacLab-ml2`，分支
  `mainline2-arc`（已推到 fork `2047767028-lang/IsaacLab`）。新 session 在这个目录里启动，
  读到的 CLAUDE.md 才是最新的（主检出 `/home/pk/IsaacLab` 的 CLAUDE.md 停在 9/2）。worktree 若被清理：
  `git worktree add ~/IsaacLab-ml2 mainline2-arc`。
- **环境**：conda `isaaclab`（Isaac Lab 2.3.2），`PY=/home/pk/miniconda3/envs/isaaclab/bin/python`；
  本机 RTX 4080 12GB，`num_envs=10` 一组 300 次约 7~14 分钟；跑之前 `nvidia-smi` 确认卡是空的。
- **跑一组**：照 `docs/arc_sweep_diagnosis/run_contact_hold_phase7.sh` 的写法（`setsid nohup ... &` 启动，
  日志里 `[时间] ####` 行是里程碑，`.hits` 文件是机制计数器）。试验脚本 `contact_hold_trial.py` 的旋钮：
  `CONTACT_FIX`、`GATE_TOL/GATE_MAX`、`PERTURB_ARC_STD`、`PERTURB_ARC_FREEZE_FRAC`、
  `PERTURB_ARC_PEAK_FRAC_MIN/MAX`、`PERTURB_ARC_STRETCH`、`RESEED_BASE`、`CMD_DIR`。
- **看结果**（都接 `<out_dir> <tag>...`）：`stage_funnel.py`（分阶段漏斗）、`seat_grasp_vs_release.py`
  （验收指标：闭合时偏心）、`effective_amplitude.py`（实际弧线，需要 CMD_DIR 日志）、
  `contact_hold_analysis.py`（接触几何 + 门控）、`big_arc_cause.py` / `arc_direction_dependence.py`（机制）。
- **数据**：`datasets/arc_sweep_diagnosis_runs/contact_hold/`（25 组，含 hdf5 / 日志 / hits / cmd 日志，不进
  git）；旧 job 的 26 组在上一级目录。源演示 `datasets/annotated_dataset.hdf5`。
- **服务器**（`surf2@192.168.3.254`，需有线口 `enp109s0` 在线、用 `ssh -b 192.168.3.123`）：9/2 那次
  mixed 第二种子续训的状态一直没查到，pipeline 日志在 `~/dev/mimic_enhance/isaac_downloads/logs/`。
