# Isaac Sim 5.1 + Isaac Lab 2.3.2 — MimicGen Pipeline 速查

环境: conda env `isaaclab` (py3.11) | Isaac Sim 5.1.0.0 | Isaac Lab 2.3.2 | torch 2.7.0+cu128

```bash
cd ~/IsaacLab
conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
```

## 1. 键盘遥操作采集（需人工操作，会弹窗）

```bash
python scripts/tools/record_demos.py \
  --device cpu --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard --num_demos 3 \
  --dataset_file ./datasets/my_demos.hdf5
```

键位: W/S=x  A/D=y  Q/E=z  Z/X,T/G,C/V=旋转  K=夹爪  R=重来本条
目标: 抓红块 -> 叠到蓝块 -> 抓绿块 -> 叠到红块 -> 松爪(必须)

## 2. 自动标注子任务边界（无需人工）

```bash
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu --headless --auto \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/my_demos.hdf5 \
  --output_file ./datasets/my_annotated.hdf5
```

淘汰规则: 重放后任务须成功, 且每个信号至少出现一次 True。

## 3. MimicGen 造数据（无需人工）

```bash
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu --headless --num_envs 10 \
  --generation_num_trials 1000 \
  --input_file ./datasets/my_annotated.hdf5 \
  --output_file ./datasets/my_generated.hdf5
```

会额外产出 `*_failed.hdf5`（失败样本，可丢弃）。

## 实测基线（官方 10 条源演示）

| 项 | 值 |
|---|---|
| 标注通过率 | 10/10 |
| 造数据成功率 | ~50% (20 成功 / 41 尝试) |
| 耗时 | 27s (含 Isaac Sim 启动), num_envs=10, CPU |
| 推算 1000 条 | 约 10-15 分钟 |

## 子任务切分（官方 Franka Stack, 共 4 段 / 3 个信号）

| # | object_ref | 信号 | 含义 |
|---|---|---|---|
| 1 | cube_2 | grasp_1 | 抓红块 |
| 2 | cube_1 | stack_1 | 红叠蓝 |
| 3 | cube_3 | grasp_2 | 抓绿块 |
| 4 | cube_2 | None | 绿叠红(末段无需信号) |

判据函数: `source/isaaclab_tasks/.../manipulation/stack/mdp/observations.py`
子任务配置: `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py`

## 安装踩坑记录

新版 pip 不带 setuptools, 导致 isaaclab 的依赖 flatdict==4.0.1 源码构建失败:
```bash
pip install "setuptools<81" "packaging==23.0" "wheel==0.45.1"
pip install --no-build-isolation "flatdict==4.0.1"
pip install -e source/isaaclab --no-build-isolation
pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
- setuptools>=81 移除了 pkg_resources -> 必须 <81
- packaging 须 ==23.0 (isaacsim-core 硬性要求)
- isaaclab.sh 装 torch 时卸掉 torchaudio 却没装回来
- pip 隔离构建环境会自带最新 setuptools -> 必须 --no-build-isolation

## VR 遥操作 (CloudXR + Isaac Teleop) —— 2026-08-24 **已打通**

主线一里程碑：Quest 3 通过 CloudXR 遥操作 Isaac Lab 单臂 Franka 叠方块任务，
**画面、追踪、夹爪控制全部正常，人可以真的操作机械臂**。

⚠️ 注意：这条链路**不在本机（笔记本）上**，跑在实验室服务器上，而且用的是
**Isaac Lab 3.0.0 + Isaac Teleop 新架构**，跟本仓库当前分支（2.3.2）那一套
`teleop_devices` / `OpenXRDevice` / CloudXR docker 容器**完全不是一回事**，别混着看。

### 为什么换到 3.0：旧方案的资料已经过期

2.3.2 文档说 Quest 3 / Pico 4 Ultra 需要申请 **CloudXR Early Access**——这个说法
现在已经过时了（当时按它走，卡在人工审批上）。实际情况：

- NGC 上那个 `cloudxr-js-early-access` 资源已经 **404 下架**。
- 取代它的是 `cloudxr-js`（无 early-access 后缀），**GA 公开可下**，许可证文件名就叫
  `NVIDIA_CloudXR_GA_License`，页面明写"针对 Meta Quest 2/3/3S 和 PICO 4 Ultra 优化"。
- 客户端也不用自己编译了：NVIDIA 官方开源了 `github.com/NVIDIA/IsaacTeleop`，
  **网页版客户端直接用浏览器打开** `https://nvidia.github.io/IsaacTeleop/client`。
- 代价：这套 Isaac Teleop 只支持 **Isaac Lab 3.0 Beta 及以上**，要求 Isaac Sim 6.0 +
  Python 3.12（3.0 明确放弃了 Isaac Sim 5.1 及以下）。

**结论：不要再去申请 Early Access，直接上 Isaac Lab 3.0。**

### 服务器环境（`surf2@192.168.3.254`，共享机 amax，4×RTX 5090）

```
~/dev/furniture_assembly/
├── IsaacLab/            # git clone -b release/3.0.0 （正式版，不是 beta2）
├── conda_env/           # 独立 conda 环境 py3.12（不碰 /opt/miniconda3 共享环境）
├── isaac-sim-6.0.1/     # 解压的 standalone 包，pip 装 Isaac Sim 后其实没用上
├── cloudxr_gpu1.env     # CloudXR profile（GPU 绑定，见下）
├── start_teleop.sh      # 启动脚本
└── stop_teleop.sh       # 停止脚本
```

安装命令（`isaaclab.sh -i` 走 pip 装 Isaac Sim，全程不需要 sudo）：
```bash
conda create -p ~/dev/furniture_assembly/conda_env python=3.12 pip
~/dev/furniture_assembly/conda_env/bin/pip install --upgrade pip   # 必须升级，旧pip解析不了
cd ~/dev/furniture_assembly/IsaacLab && ./isaaclab.sh -i "mimic,teleop,isaacsim"
```
实测耗时约 **3.5 小时**（服务器下行约 5MB/s，光 `isaacsim-extscache-kit` 就 5.5GB）。

防火墙（新版 CloudXR.js web 客户端只要这 3 个口，跟 2.3.2 文档那一长串完全不同）：
```bash
sudo ufw allow 49100/tcp   # 信令
sudo ufw allow 47998/udp   # 媒体流
sudo ufw allow 48322/tcp   # WSS 代理(HTTPS)
```

### 启动 + 连接

服务器端：
```bash
ssh surf2@192.168.3.254
bash ~/dev/furniture_assembly/stop_teleop.sh      # 先清理残留
setsid nohup ~/dev/furniture_assembly/start_teleop.sh > ~/dev/furniture_assembly/teleop.log 2>&1 < /dev/null &
disown
# 日志出现 "OpenXR handles not yet available (waiting for XR session)" 即就绪
```
Quest 3 端：浏览器打开 `https://nvidia.github.io/IsaacTeleop/client` → Server IP 填
`192.168.3.254` → 点页面上的 "accept cert" 链接接受自签名证书 → 回来点 **Connect**
→ 点 **Run**。

### 操作方式（`IsaacContrib-Stack-Cube-Franka-IK-Abs`，源码逐行核实过）

| 输入 | 作用 | 依据 |
|---|---|---|
| 右手手柄位置/朝向 | 直接驱动末端执行器（**绝对位姿映射**，1:1空间对应） | `se3_retargeter.py:256` 读 `GRIP_POSITION`/`GRIP_ORIENTATION` |
| 右手扳机 Trigger | >0.5 夹爪闭合，松开张开 | `gripper_retargeter.py:96-101`，`controller_threshold=0.5` |
| 左手手柄 | **无作用**（pipeline 只连了 `ControllersSource.RIGHT`） | `stack_ik_abs_env_cfg.py` |

姿态有个 `target_offset_roll=90°` 的偏置，手柄朝向和夹爪朝向差 90 度是设计如此。
任务：`cube_1`=蓝 `cube_2`=红 `cube_3`=绿 → 抓红叠蓝 → 抓绿叠红。
任一方块掉出桌面（低于 -0.05m）直接判 episode 结束。

### ⚠️ 打通过程中踩的坑（都很隐蔽，重来一遍必踩）

**① 多 GPU 机器上，CloudXR 编码器和 Isaac Sim 渲染器会跑到不同的物理卡上 → 头显里
是雪花/黑屏**（最难查的一个，症状极具误导性：连接成功、会话正常、编码器日志里帧率
耗时都正常，就是画面是垃圾）。

根因：**Vulkan / CUDA / nvidia-smi 是三套互不相同的 GPU 编号体系**。本机实测：

| Vulkan 编号 | 对应 nvidia-smi 编号 |
|---|---|
| 0 | GPU2 |
| 1 | **llvmpipe（CPU软件渲染器，根本不是显卡）** |
| 2 | GPU0 |
| 3 | GPU1 |

CloudXR 默认 `gpuIndexVulkan: -1`（自动）会选 Vulkan 0 号 = 物理 GPU2；而 Isaac Sim
渲染在别的卡上，编码器去读一张没有画面的卡的显存，编出来就是雪花。

**排查方法**：用 `deviceUUID` 做映射，别信编号：
```bash
vulkaninfo 2>/dev/null | grep -E "^GPU[0-9]+:|deviceName |deviceUUID "
nvidia-smi --query-gpu=index,uuid --format=csv    # 两边 UUID 对照
```
**解法**：`NV_CXR_GPU_INDEX_VULKAN=<vulkan编号>` 写进 CloudXR env profile。
注意 **`NV_CXR_GPU_INDEX_VULKAN` 和 `NV_CXR_GPU_INDEX_CUDA` 只能二选一**，同时设会
直接报 `Only one of gpu-index-vulkan and gpu-index-cuda may be set at a time`。

**② 同一张 GPU 被多个 Vulkan ICD 重复枚举**（Isaac Sim 会警告"leads to instability
or crash"）。本机 `/etc/vulkan/icd.d/nvidia_icd.json` 和
`/usr/share/vulkan/icd.d/nvidia_icd.json` 内容相同、都被加载，导致每张卡数两遍
（9个设备）。**解法**：进程级环境变量只加载一个（不改共享机系统文件）：
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json   # 注意是旧变量名
```
⚠️ Ubuntu 22.04 的 Vulkan 加载器**只认旧的 `VK_ICD_FILENAMES`**，新的
`VK_DRIVER_FILES` 设了没反应（实测）。设完设备列表从 9 个变 4 个，**编号也会跟着变**，
必须重新做一次 UUID 映射。

**③ Isaac Lab 的 `--device cuda:N` 不会把渲染器也钉到那张卡**（详见下面 bug 记录），
必须自己额外加 `--kit_args "--/renderer/multiGpu/activeCudaGpus=N,"`（末尾逗号不能省，
是为了让该设置保持字符串类型）。不加的话渲染器会跨卡，日志里会出现
`Usdrt Hydra CUDA Peer Memory Copies from device[0] to device[2] is NOT possible`，
而且容易在别人占满的卡上 OOM 崩掉。

**④ CloudXR 有独立于 Omniverse 之外的第二个 EULA**，首次运行会交互式问 `y/N`，
在 `nohup` 后台跑时 stdin 是空的直接失败退出。接受一次后写标记文件
`~/.cloudxr/run/eula_accepted`，之后不再问。**Isaac Lab 目前没有免交互接受的开关**
（详见下面 bug 记录）。

**⑤ 共享服务器的显存是动态变化的，不能只看一次快照**。踩过：查的时候 GPU0 只占
3.6GB，等装完开始跑，别人一个 22GB 的进程起来了，直接 OOM 崩溃
（`LLVM ERROR: out of memory` / `VkResult: ERROR_OUT_OF_DEVICE_MEMORY`）。
每次启动前先 `nvidia-smi` 挑当前最空的卡。

**⑥ `pkill -f "teleop_se3_agent"` 会把自己所在的 ssh 远程 bash 一起杀掉**
（远程执行的整条命令行里含有这个字符串，`-f` 匹配整行 → 自匹配），表现为 ssh
莫名返回 255。**解法**：把启停逻辑写进脚本文件再执行，命令行里就不含该字符串了；
或者在 pattern 里拆字符串（`"teleop_se3_ag""ent"`）。

### 下一步

用 `scripts/tools/record_demos.py` 配同样的参数录真实演示数据（替代 RUNBOOK 第1节的
键盘录制），补上"自己遥操作那次录制损坏"的缺口。官方教程提醒：**用 Ctrl-C 退出后必须
手动清理残留的 CloudXR 进程**，否则下次启动会报 `XR_ERROR_INSTANCE_LOST`：
```bash
pkill -KILL -f '[i]saacteleop.cloudxr.runtime'
```
