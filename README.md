# AgiBotSim

## 依赖

- conda 环境（如 `agibotsim`）：`numpy`，`pinocchio`（`conda install -c conda-forge pinocchio`）
- Isaac Sim：仅标准库 + isaacsim（无需在 Isaac 中安装 pinocchio）
- r3kit：在Submodules目录下克隆 git clone git@github.com:dadadadawjb/r3kit.git 并安装

## 双进程：IK -> 共享内存 -> Isaac

1. **IK 进程**（conda）：伪状态（含基座 xy/朝向）-> Pinocchio 反解 -> 基座位姿 + 46 关节角写入 shm  
   `python Scripts/ik_to_shm.py`

2. **Isaac 进程**（bat）：从 shm 读位姿+关节，插值后 `set_world_pose` + `apply_action`  
   将 `start.bat` 中 `SCRIPT_NAME` 设为 `isaac_from_shm.py` 后运行。  
   URDF 以 **fix_base=False** 导入，支持整体移动。

**圆周行走**：在 `fake_states_generate.py` 中 `POSE_MODE = "circle"`（默认），并设置 `CIRCLE_RADIUS`、`CIRCLE_OMEGA`、`CIRCLE_Z`。Isaac 端通过 `POSE_LERP_TAU` / `JOINT_LERP_TAU` 控制插值平滑度，减小对物理的冲击。