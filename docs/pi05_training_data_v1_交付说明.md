# pi0.5 训练数据 v1：交付说明

> 主线二第二层验证的训练数据生产（CLAUDE.md"主线二·第二层验证启动"）。
> 这份文件的目的是"数据备好了、在哪、怎么用"，不是曲线分析——曲线/成功率分析见
> `docs/接触锚定扰动增广_实验记录_v2_arc.md`（幅度扫描）；这次是固定配置的生产跑。

## 1. 数据在哪

```
datasets/pi05_training_data_v1/
├── baseline/
│   ├── generated.hdf5          # 380 条成功demo（训练用这个）
│   ├── generated_failed.hdf5   # 670 条失败demo（保留，未清理）
│   └── log.txt / result.txt
└── arc_1p2cm/
    ├── generated.hdf5          # 358 条成功demo（训练用这个）
    ├── generated_failed.hdf5   # 692 条失败demo（保留，未清理）
    └── log.txt / result.txt
```

⚠️ 这个目录在 `.gitignore` 里（`/datasets/`），**不会随 `git clone`/`git pull` 带走**，
需要用户自己拷贝/rsync到实验室服务器。

## 2. 两组配置

| | baseline | arc_1p2cm |
|---|---|---|
| `PERTURB_STD` | 0 | 0 |
| `PERTURB_ARC_STD` | 0 | 0.012（1.2cm） |
| `PERTURB_ARC_FREEZE_FRAC` | — | 0.3 |
| `PERTURB_SEED` | 1 | 1 |

任务：`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-Perturbed-v0`（新增，见下"过程"节）。
`num_envs=10`，`PERTURB_FIXED_ATTEMPTS=1`（固定1050次尝试，不是"凑够N个成功"）。

## 3. 最终数字

| | baseline | arc_1p2cm |
|---|---|---|
| 成功demo数 | **380** | **358** |
| 失败demo数 | 670 | 692 |
| 总尝试数 | 1050（核实无误） | 1050（核实无误） |
| 成功率 | 36.2% | 34.1% |
| 耗时 | 7270.5秒（约2.02小时） | 7720.2秒（约2.14小时） |
| 磁盘占用 | 24GB（成功+失败合计） | 24GB（成功+失败合计） |

两组都超过用户定的300条目标（官方建议100~200条的1.5倍上限），不需要额外补采。
总耗时约4.16小时，12小时预算内，未触及预算上限。

**磁盘**：两组合计47GB，当前磁盘剩余39GB（89%已用）——数据已生产完，不需要再占用磁盘，
但提醒一下：如果之后还要在这台机器上生成更多带图像的数据，剩余空间不算宽裕。

## 4. 数据完整性核查

- **脚本自带核验**：两组都 `n_success + n_failed == 1050`（重新打开hdf5数demo数确认，
  不是只信进程退出码）。
- **图像完整性**（这次新加的检查项，之前纯状态扫描没有）：每组抽样10个成功demo的
  `table_cam`/`wrist_cam`首帧，检查是否退化（全零/全同值）——**两组均无异常**。
- **我又独立复核了一遍**（不只信脚本自己的结果）：
  - 直接重新打开两组的 `generated.hdf5`/`generated_failed.hdf5` 数demo数，
    结果跟脚本记录完全一致（380/670、358/692）。
  - 独立抽查了每组第一条成功demo的 `table_cam`/`wrist_cam` **首帧和中间帧**（脚本自带检查
    只查了首帧），像素统计量都在合理范围（比如baseline的`table_cam`首帧 mean=141.7,
    std=63.3, min=1, max=252——有明显的明暗分布，不是单一颜色）：

    | | table_cam mean/std | wrist_cam mean/std |
    |---|---|---|
    | baseline 首帧 | 141.7 / 63.3 | 103.5 / 27.3 |
    | baseline 中间帧 | 145.5 / 64.5 | 103.4 / 25.8 |
    | arc 首帧 | 142.7 / 62.7 | 104.6 / 26.8 |
    | arc 中间帧 | 146.2 / 64.1 | 101.7 / 27.8 |

  - 确认每条demo的 `obs` 组里同时有状态量（`eef_pos`/`joint_pos`/`cube_positions`等）
    和图像（`table_cam`/`wrist_cam`，`(200,200,3)` uint8），两者都在，不是只有一种。

## 5. 过程中的一个bug（已修复，供参考）

冒烟测试阶段发现：视觉版任务（`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`，
之前扫描用的纯状态任务的带图像版）**不认 `PERTURB_STD`/`PERTURB_FIXED_ATTEMPTS`**——
这两个环境变量只在之前扫描专用的 `FrankaCubeStackIKRelPerturbedMimicEnvCfg` 里接了线，
视觉版任务是完全独立的类，压根没读这两个变量（`PERTURB_ARC_STD`不受影响，
它是写在通用MimicGen代码里的，不绑定某个env cfg）。实测现象：设了
`PERTURB_FIXED_ATTEMPTS=1`+目标20次，结果跑到快30次还没停——因为
`generation_guarantee`还是硬编码的`True`，在"凑够20个成功"模式下运行，
不是"固定跑20次"。**更隐蔽的是**：如果不修，"baseline组`PERTURB_STD=0`"这个配置会
静默失效，实际上还是套着任务自带的固定`std=0.02`关节噪声，"基线"就不干净了。

已修：新建 `FrankaCubeStackIKRelVisuomotorPerturbedMimicEnvCfg`
（`source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_visuomotor_perturbed_mimic_env_cfg.py`），
接上跟纯状态版一样的三个环境变量，注册成新的gym id
`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-Perturbed-v0`（commit `02ed4b14e`）。
踩到bug时冒烟测试跑的是错误配置，**已经把那次的输出删了，没有用来生产正式数据**——
正式生产用的从头到尾都是修复后的task id。

## 6. 怎么用

拷到服务器后，两个 `generated.hdf5` 就是可以喂给数据转换脚本的原始产物
（robomimic hdf5格式，含图像观测）。如果 OpenPI/pi0.5 训练流程要 LeRobot 格式，
转换脚本目前仓库里没有现成的（CLAUDE.md"发现②"已经记录这个缺口），需要另外搭。
`generated_failed.hdf5` 是失败demo，训练用不上，保留是为了以后万一要分析失败模式，
不需要现在处理。
