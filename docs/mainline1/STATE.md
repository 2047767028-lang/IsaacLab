# 主线一 状态页（家具装配 / VR 遥操作 / 基准调研）

> **你应该在**：目录 `~/IsaacLab-ml1`，分支 `mainline1-furniture`（Isaac Lab 3.0 基线）。对不上就停下来告诉我。
> 本页由主线二的 session 于 2026-09-04 从旧 CLAUDE.md 搬来，**内容截至 2026-09-02**；主线一的 session 接手后
> 请把它改成自己的一页状态，并把 9/3~9/4 的调研（本分支 `docs/` 下五份中文调研文档、`git log`）纳入。

### 主线一：双臂 Franka 家具装配（原始项目目标，当前焦点）

**当前进度**：目前跑通的是官方单臂 `Isaac-Stack-Cube-Franka-IK-Rel-v0`（叠方块任务）的
record → annotate → generate 全流程，用的是**官方自带的 10 条源演示**，不是自己遥操作
采出来的数据；也还是**单臂**，不是双臂家具装配。细节见 `RUNBOOK_mimic.md`。
双臂 Franka 环境和家具装配任务/资产目前在本项目里还没有配置。
自己遥操作那次（`my_demos.hdf5`）文件损坏/未录成，还没补录。

**2026-08-27：VR 遥操作已打通** ✅ 用户实测可正常操作（画面正常、右手手柄驱动机械臂、
扳机控夹爪）。**注意跑在实验室服务器上，不是本机**：`surf2@192.168.3.254` 的
`~/dev/furniture_assembly/`，用的是 **Isaac Lab 3.0.0 + Isaac Teleop 新架构**，跟本仓库
当前分支（2.3.2 的 `teleop_devices`/`OpenXRDevice`/CloudXR docker 容器那一套）**完全不是
一回事**，别混着看。完整步骤、连接方式、操作按键、6 个踩坑记录见 `RUNBOOK_mimic.md`
"VR 遥操作 (CloudXR + Isaac Teleop)"节；服务器配置文件备份在
`docs/vr_teleop_server_config/`。

⚠️ **一条已作废的旧结论**：2.3.2 文档说 Quest 3/Pico 4 需要申请 CloudXR Early Access
——已过时，NGC 上那个资源已 404 下架，现在 `cloudxr-js` 是 GA 公开可下，客户端直接用
浏览器打开 `https://nvidia.github.io/IsaacTeleop/client`。**不要再去申请 Early Access。**

**向上游提了四个 PR**（都已核实存在于 upstream `develop`，非本地环境问题；状态 2026-09-02
核实，随时用 `gh pr list --repo isaac-sim/IsaacLab --author 2047767028-lang --state all` 重查）：
- #7381 多 GPU 机器上 CloudXR 编码器和渲染器跑到不同物理卡 → 头显里是雪花。**已合并**
  （2026-08-31）。
- #7380 CloudXR EULA 无法免交互接受 → headless/CI 启动必失败。分支
  `fix/cloudxr-eula-noninteractive`。AntoineRichard 的三条意见（replay 路径漏接、测试隔离、
  文档）8/29 已改完（`7c48a40cf`）；#7381 合并后跟 develop 冲突——两个 PR 在
  `session_lifecycle.py` 同一位置各加了一段模块级辅助函数，9/2 用 **merge develop** 的方式
  解决（`c4711125b`，两段纯新增全保留；不 rebase 是为了不 force-push 改写 PR 上的 review
  历史），`test_cloudxr_lifecycle.py` 33 个用例在 `ISAACLAB_CXR_ACCEPT_EULA` 未设/`1`/`0`
  三种取值下全过，已在 PR 回复。现在等维护者审批。
- #7433 `DataGenConfig.max_num_failures` 全仓库 18 处赋值但从未被读取（就是主线二 2.1 节
  踩的那个坑）→ 让它真正兜底。分支 `fix/mimic-max-num-failures`。两条 bot 意见已处理：
  `generation_guarantee` 门控已加（`3bcd74c09`，固定尝试次数模式不受 cap 影响）；"新文件
  license 头应该是 BSD"是**误报**——`.pre-commit-config.yaml` 对 `source/isaaclab_mimic/`
  强制 Apache-2.0 头，隔壁 `mimic_test_utils.py` 就是同款，已在 PR 里回复。等
  kellyguo11/peterd-NV 审批。
- #7434 `cubes_stacked` 只看瞬时几何构型，方块下落途中经过"叠好"的高度也算成功，而 Mimic
  生成器按"任意帧曾成功"累计，一帧误判就把整条 episode 当成功 demo 写进数据集 → 加线速度
  门控（默认 0.05 m/s，可传 `None` 关掉）。分支 `fix/cubes-stacked-at-rest`。ooctipus 要求
  删测试文件，kellyguo11 8/31 接受"不用 Kit 能快跑就留着"，9/2 删掉 `AppLauncher` 前导
  （`282d9cca0`），12 个用例 0.68 秒无 Kit 通过，已在 PR 回复。等审批。

前两个的核实过程、证据、代码定位见 `docs/upstream_bug_findings_xr_teleop.md`。

## 本分支上的文档

- `RUNBOOK_mimic.md`（VR 遥操作章节在此）、`docs/vr_teleop_server_config/`（服务器配置备份）、
  `docs/upstream_bug_findings_xr_teleop.md`（#7380/#7381 的证据）。
- 9/3~9/4 的基准与资产调研：`docs/` 下的中文文档（`git log --oneline release/3.0.0-beta2..HEAD` 可见）。
