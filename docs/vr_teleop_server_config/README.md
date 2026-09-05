# VR 遥操作服务器配置文件备份

这三个文件的**真身在服务器上**：`surf2@192.168.3.254:~/dev/furniture_assembly/`。
这里存一份是为了服务器重装后能直接复现，不是本仓库的运行时代码
（本仓库当前分支是 Isaac Lab 2.3.2，这套配置是给服务器上那份 3.0.0 用的）。

背景、踩坑、连接步骤见 `RUNBOOK_mimic.md` 的"VR 遥操作 (CloudXR + Isaac Teleop)"节。

| 文件 | 用途 |
|---|---|
| `start_teleop.sh` | 启动 XR 遥操作会话，把物理/渲染/CloudXR 全钉在物理 GPU1 |
| `stop_teleop.sh` | 停掉会话和残留的 CloudXR runtime |
| `cloudxr_gpu1.env` | CloudXR device profile，含 GPU 绑定和单 ICD 设置 |

⚠️ **换机器/换卡就必须重新算 GPU 索引**。`cloudxr_gpu1.env` 里的
`NV_CXR_GPU_INDEX_VULKAN=2` 只对这台机器 + 这个 `VK_ICD_FILENAMES` 设置成立。
Vulkan 编号和 nvidia-smi 编号是两套体系，重新映射的方法：

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
vulkaninfo 2>/dev/null | grep -E "^GPU[0-9]+:|deviceUUID "
nvidia-smi --query-gpu=index,uuid --format=csv
# 按 UUID 对照，找出目标物理卡在 vulkaninfo 里排第几个（从0数）
```
