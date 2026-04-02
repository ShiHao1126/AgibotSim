import os
import sys
import time
import math
import asyncio
import numpy as np
from scipy.spatial.transform import Rotation

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, UsdLux
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from Utils.shm import shm_read
from Utils.transform import quat_project_to_xy_plane

URDF_PATH = os.path.join(ROOT_DIR, "Assets", "GenieG2", "G2_t1_crs_omnipicker.urdf")
NUM_JOINTS = 46


async def main():
    # world initialization
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight")).CreateIntensityAttr(500.0)

    # URDF import
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.fix_base = False
    import_config.import_inertia_tensor = True
    import_config.collision_from_visuals = True
    import_config.create_physics_scene = False
    import_config.parse_mimic = True
    result, prim_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=URDF_PATH, import_config=import_config, dest_path="")
    if not result or not prim_path:
        return

    # robot initialization
    robot = SingleArticulation(prim_path=prim_path, name="genie_robot")
    world.scene.add(robot)
    await world.reset_async()
    await world.play_async()
    robot.initialize()


    def step(dt: float):
        data = shm_read()
        if data is None:
            return
        xyz, quat, joints = data
        target_pos = np.array([xyz], dtype=np.float64)
        target_quat = quat_project_to_xy_plane(np.array(quat, dtype=np.float64))
        target_joints = np.array(joints, dtype=np.float64)

        # set world pose
        xyz = np.asarray(xyz, dtype=np.float64).ravel()
        robot.set_world_pose(
            Gf.Vec3f(float(xyz[0]), float(xyz[1]), 0.0),
            target_quat.astype(np.float32),
        )
        # set joint positions
        robot.set_joint_positions(target_joints.astype(np.float32))

    world.add_physics_callback("server_sim", step)


if __name__ == "__main__":
    asyncio.ensure_future(main())
    while True:
        simulation_app.update()
        time.sleep(0.001)
