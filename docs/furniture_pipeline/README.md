# 家具装配资产 pipeline —— 已打通（2026-09-04）

**目标**：把一个主流格式的 3D 家具模型变成 USD，导入 Isaac Lab，跑通物理。
**状态**：✅ **OBJ → USD → Isaac Lab → 物理稳定沉降，端到端跑通。**

工作副本与产出资产在 **`/home/pk/furniture_assembly/`**（worktree 可能被删，所以放在外面）。
本目录是脚本的版本控制副本。

## 1. 选的资产：FurnitureBench `square_table`（MIT 许可）

选它的理由：许可干净（MIT）、零件已按装配单元拆分、**而且真有装配特征**。
同时它对得上三件事：FurnitureBench 最简任务（one-leg table）、IROS 2026 挑战赛的
UTTER 桌（1 桌面 + 4 腿）、以及我们之前那个桌腿 demo。

### 实测几何（`measure_obj.py`）
| 零件 | 尺寸 (x × y × z) | 顶点 / 面 |
|---|---|---|
| `square_table_top` | **162.5 × 31.2 × 162.5 mm** | 12090 / 24128 |
| `square_table_leg1` | **30.0 × 87.5 × 30.0 mm** | 5633 / 11224 |
| leg2 / leg3 / leg4 | 同上（各自略有差异） | ~5.5k~6k / ~11k~12k |

→ 整张桌约 16 cm 见方，**约为 IKEA UTTER 儿童桌（~48×48×45 cm）的 1/3**。
这就是"玩具尺度"的精确含义。

### 装配特征确认（`slice_obj.py` / `find_holes.py`）
**这些零件是真能拧到一起的**，不是简单的几何切分：

- **桌腿**：30×30 mm 方杆（约 58 mm）+ 一端约 24 mm 的圆形凸台，
  半径在 **7.82 ~ 12.19 mm** 之间起伏，且该区域每个切片有 950~1030 个顶点
  —— 螺纹面的典型特征。
- **桌面**：中层切片聚类出 **4 个孔**，各 284 个顶点，中心在四角
  (±56, ±50/63) mm，孔口约 29 × 23 mm —— 与桌腿凸台匹配的螺纹孔。

> 对比：IKEA-Manual 的 102 件**没有**这类配合特征（论文明说
> "we do not model connectors"）。FurnitureBench 有，因为它本来就是为
> 3D 打印和真实拧装设计的。

## 2. 转换：`convert_square_table.sh`

```bash
bash convert_square_table.sh     # 约 4 分钟，5 个零件
```

调用 Isaac Lab 自带的 `scripts/tools/convert_mesh.py`。

**关键选择：`--collision-approximation sdf`。**
这不是偏好问题 —— `convexHull` / `convexDecomposition` 会把螺纹和孔**全部抹平**，
装配在物理上就不可能发生。SDF 是唯一能保住配合特征的碰撞近似。

质量按 3D 打印件（PLA，大部分中空）估：桌面 0.15 kg，每条腿 0.03 kg。

## 3. 验证转换结果：`inspect_usd.py`

**转换器返回 0 不等于物理属性真的写进去了**，必须读回 stage 核对。

```bash
export LD_LIBRARY_PATH="<isaacsim>/extscache/omni.usd.libs-*/bin:/home/pk/miniconda3/envs/isaaclab/lib:$LD_LIBRARY_PATH"
python3 inspect_usd.py /home/pk/furniture_assembly/assets/square_table/*.usd
```

实测结果（已核对）：
- `PhysicsRigidBodyAPI` + `PhysicsMassAPI`，`physics:mass` = 0.15 / 0.03 kg ✅
- `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI`，**`physics:approximation = sdf`** ✅
- 网格未简化（桌面 12090 点 / 24128 面，与源 OBJ 一致）✅
- Z-up，meters/unit = 1.0 ✅
- 材质与贴图保留 ✅

## 4. 物理验证：`verify_pipeline.py`

```bash
python3 verify_pipeline.py --headless --yup-fix
```

把 5 个零件从 0.25 m 高处丢到地面，跑 400 步（dt=1/120，约 3.3 s），
检查：数值有限、速度收敛、未穿透地面、最后 60 步位移 < 10 mm。

**结果：`PIPELINE: PASS`（5/5 零件全部通过）**

## 5. ⚠️ 踩到的坑：Y-up / Z-up 不一致（重要）

**`convert_mesh.py` 不会旋转几何。** FurnitureBench 的 OBJ 是 **Y-up**（Isaac Gym 约定），
而 USD stage 是 **Z-up** —— 转换后零件全是**躺倒/立起**的错误姿态。

这个坑很隐蔽，因为**物理照样跑通、验证照样 PASS**，只有量沉降高度才看得出来：

| | 未修正 | 加 `--yup-fix` | 应有值 |
|---|---|---|---|
| 桌面沉降 z | **0.0813 m**（= 162.5/2 mm，**立在边上**） | **0.0157 m** | 0.0156 m（= 31.2/2，平躺）✅ |
| 三条腿 z | 0.0144 ~ 0.0156 m | 0.0150 ~ 0.0156 m（侧躺，= 30/2） | ✅ |
| leg3 z | — | **0.0563 m** | 正好等于其网格局部原点到底端的距离（−0.056）→ **立着的**，物理合理 ✅ |

**当前是在 spawn 时加四元数 (w,x,y,z)=(0.7071, 0.7071, 0, 0)（绕 X +90°）绕过的。**
作为一次性验证可以，但**资产库不该这样**——正确做法是把旋转烘焙进 USD，
让资产本身就是 Z-up 正确姿态。**这是下一步要做的第一件事。**

## 6. 环境备忘

- Isaac Sim **5.1.0** + isaaclab **0.54.2**，conda env `/home/pk/miniconda3/envs/isaaclab`
- ⚠️ `import isaaclab` 是 editable 安装，**指向主仓库 `/home/pk/IsaacLab/source/isaaclab`**，
  不是本 worktree。跑脚本要用主仓库的 `scripts/tools/`，避免版本错配。
- 无头运行必须设 `OMNI_KIT_ACCEPT_EULA=YES`，否则卡在 EULA 交互提示。
- Kit 会劫持 stdout，脚本的结果要**写文件**，不要靠 print。
- GPU：RTX 4080 Laptop 12 GB，够用。

## 7. 下一步

1. **把 Y-up→Z-up 旋转烘焙进 USD**，让资产姿态本身就对（§5）
2. 加 Franka，做抓取 + 插孔，验证 **SDF 碰撞下螺纹配合能不能真的装进去**
   —— 这是整个 benchmark 最核心的物理问题，也是最可能翻车的地方
3. 接 Quest 遥操作（服务器上 IsaacTeleop 链路已通）
4. 真实尺度资产（本批是 1/3 玩具尺度，撑不起"real-scale"卖点）
