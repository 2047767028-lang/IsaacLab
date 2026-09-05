# Isaac Lab 上游问题核实记录（XR 遥操作 / 多 GPU）

发现时间：2026-08-24 ~ 08-26，在 `surf2@192.168.3.254`（共享机 amax，4×RTX 5090）
上打通 Quest 3 + CloudXR 遥操作 Isaac Lab 3.0 的过程中踩到。

## 提交结果（2026-08-27）

| 问题 | PR | 链接 |
|---|---|---|
| C（EULA 免交互） | #7380 | https://github.com/isaac-sim/IsaacLab/pull/7380 |
| A + B（多 GPU 设备对齐） | #7381 | https://github.com/isaac-sim/IsaacLab/pull/7381 |

两个 PR 都基于 `develop`，带 DCO 签名，本地跑过完整 pre-commit 全绿（含 ruff、
codespell、changelog fragment 校验），并在服务器上用真实 Quest 3 端到端验证过
（纯补丁代码、零手工 workaround，画面正常且可正常遥操作）。

未勾选的两个 checklist 项已在 PR 里说明原因：①复现需要"Vulkan 与 CUDA 枚举顺序不一致
的多 GPU 主机 + 连着的头显"，无法做成 CI 能跑的测试；②`CONTRIBUTORS.md` 要求真实姓名，
留给用户自己决定是否添加。

**核实基线**：全部对照上游 `develop` 分支最新代码确认仍然存在（不是我们装的 3.0.0
才有的问题），并检索过已有 issue/PR 排除重复。PR 目标分支是 `develop`
（贡献文档写的 `main` 已过时，`main` 上没有 `isaaclab_teleop`；最近合并的 PR 全部
以 `develop` 为 base）。

---

## A. CloudXR runtime 不跟随仿真 GPU → 多卡机器上头显里是雪花/黑屏

**状态**：✅ 端到端实测确认，上游未修复，无重复 issue。

**位置**：`source/isaaclab_teleop/isaaclab_teleop/session_lifecycle.py`
`_ensure_cloudxr_runtime()`（develop 第 1286 行起）：

```python
self._cloudxr_launcher = _CloudXRLauncher(
    install_dir=str(Path.home() / ".cloudxr"),
    env_config=self._cloudxr_env_file,
    accept_eula=False,
)
```

构造时**不传任何 GPU 绑定信息**，CloudXR runtime 于是回落到自动选择
（服务端日志：`gpuIndexVulkan: -1` / `gpuIndexCuda: -1`），选中 Vulkan 枚举里的第 0
个设备。

**为什么这在多卡机器上必然出错**：Vulkan、CUDA、nvidia-smi 是三套互不相同的编号
体系。本机实测映射：

| Vulkan 编号 | 对应 nvidia-smi |
|---|---|
| 0 | GPU2 |
| 1 | llvmpipe（CPU 软件渲染器，根本不是显卡） |
| 2 | GPU0 |
| 3 | GPU1 |

CloudXR 自动选 Vulkan 0 = 物理 GPU2，而 Isaac Sim 渲染在别的卡上 → 编码器读一张
没有画面的卡的显存。

**症状极具误导性**：客户端连接成功、`IsaacTeleop session started`、编码器日志里
`GpuEndToEncodeEnd 7.1ms` 等帧率统计一切正常，**只有画面是雪花**。完全不报错，
排查时很容易误判成网络或客户端问题。

**验证**：把 CloudXR 钉到 Isaac Sim 所在的那张卡之后，画面立刻正常，人可以实际
遥操作机械臂完成叠方块任务。

**修复方向**：从仿真设备推导索引传给 launcher。注意 CloudXR **只接受两个索引里的
一个**，同时设会直接失败：
`Only one of gpu-index-vulkan and gpu-index-cuda may be set at a time`。
用 `NV_CXR_GPU_INDEX_CUDA` 更合适——它和 Isaac Lab 已有的 `--device cuda:N` 是同一
套编号，不需要去解析 Vulkan 枚举顺序。

---

## B. `--device cuda:N` 不会把渲染器钉到那张卡（与代码注释自相矛盾）

**状态**：✅ 代码 + 运行时探针双重确认，上游未修复。属于"注释声明的行为没有兑现"。

**位置**：`source/isaaclab/isaaclab/app/app_launcher.py`

第 1135-1138 行**无条件**声称渲染器设备会被选择：
```python
# ``/physics/cudaDevice`` is resolved by CUDA, so the masked index is correct there.
# ``activeGpu`` is deliberately left unset; the renderer device is selected in
# :meth:`_resolve_kit_args` instead.
launcher_args["physics_gpu"] = self.device_id
```

但第 1302 行实际有门槛：
```python
if launcher_args.get("multi_gpu") is False:
    argument = f"--/renderer/multiGpu/activeCudaGpus={self.device_id},"
```

而 `multi_gpu` 全文件**只在第 1124 行被赋值**，且位于
`if "distributed" in launcher_args and launcher_args["distributed"]:` 分支内。
`--multi_gpu` 和 `--distributed` **都不是命令行参数**（`add_argument` 里没有），
`_sim_app_config` 是按键取交集构建的（第 807 行）不会填默认值。

⇒ 非分布式运行时 `multi_gpu` 这个键根本不存在，`.get()` 返回 `None`，
`None is False` 为假，渲染器**永远不会被钉住**。

**运行时探针实测**（读 carb settings 真实值）：

| 命令 | `/physics/cudaDevice` | `/renderer/multiGpu/activeCudaGpus` | `/renderer/multiGpu/enabled` |
|---|---|---|---|
| `--device cuda:1` | `1` ✅ | **`None`** ❌ | `True` |
| `--device cuda:1 --kit_args "--/renderer/multiGpu/activeCudaGpus=1,"` | `1` | `'1,'` ✅ | `True` |

即：物理引擎在 GPU1，渲染器跨全部 4 张卡。日志里可观察到后果：
`Usdrt Hydra CUDA Peer Memory Copies from device[0] to device[2] is NOT possible as
peer access is disabled`，以及在别人占满的卡上 OOM 崩溃
（`LLVM ERROR: out of memory` / `VkResult: ERROR_OUT_OF_DEVICE_MEMORY`）。

**相关上游改动**（不是重复，范围不同）：
- PR #7057「Select the renderer device by CUDA index」(2026-08-15 合并) —— 把
  `activeGpu` 换成 `activeCudaGpus`，**范围明确限定在分布式多卡训练**
  （`train_multigpu.py` / rank / `CUDA_VISIBLE_DEVICES` 掩码），那个 `multi_gpu`
  门槛是为它服务的，没有覆盖单进程场景。
- PR #7347「Pin the OVRTX render product to the renderer's CUDA device」
  (2026-08-26 合并) —— 同类"多卡设备错配"问题，但走的是 OVRTX 相机路径，
  跟 Kit 渲染器选择和 CloudXR 都无关。

**与 A 的关系（决定了修复方案）**：A 的修复需要"渲染器在哪张卡"有确定答案，而 B 正是
让这个答案不确定的原因 —— 所以两者必须一起修，否则 A 单独修也定位不到正确的卡。

**采取的修法（把改动范围收敛到 XR，不动通用默认）**：只在 `--xr` 开启时额外钉住渲染器
（`if launcher_args.get("multi_gpu") is False or self._xr:`）。理由：XR 推的是单一立体
swapchain，必须由 CloudXR 合成器导入，渲染器和合成器必须在同一张卡上；而非 XR 的单进程
场景保持原样，不去推翻另一位贡献者刚合并的设计取舍。

---

## C. CloudXR EULA 无法免交互接受 → headless / CI / nohup 启动必失败

**状态**：✅ 代码确认，上游未修复。

**位置**：同 `_ensure_cloudxr_runtime()`，`accept_eula=False` 是**硬编码**的。

CloudXR 有一个独立于 Omniverse 之外的第二份 EULA。`accept_eula=False` 时
`isaacteleop` 会走交互式 stdin 询问：

```
NVIDIA CloudXR EULA must be accepted to run. View: <...>
Accept NVIDIA CloudXR EULA? [y/N]:
```

后台/CI 环境没有 stdin，直接：
```
EULA not accepted. Exiting.
RuntimeError: CloudXR EULA was not accepted; cannot start the runtime
```

**上游依赖库本身是支持免交互的**：`isaacteleop.cloudxr.CloudXRLauncher` 有
`accept_eula: bool` 参数，还注册了 `--accept-eula` 命令行开关
（`add_accept_eula_argument`，注释原文 "e.g. for CI or containers"）——
**但 Isaac Lab 这一层完全没有把它暴露出来**，永远传 `False`。

对比：Omniverse 那份 EULA 有官方文档化的 `OMNI_KIT_ACCEPT_EULA=YES` 环境变量，
CloudXR 这份没有任何对应通道。

**修复方向**：加一个 CLI 开关（如 `--accept_cloudxr_eula`）或环境变量
（如 `ISAACLAB_CXR_ACCEPT_EULA=1`），透传给 launcher。附近已有同类环境变量先例：
`ISAACLAB_CXR_SKIP_AUTOLAUNCH`。
