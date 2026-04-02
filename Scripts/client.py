import os
import sys
import time
from tap import Tap
import numpy as np
from typing import List
import pinocchio as pin
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from Submodules.r3kit.r3kit.devices.tracker.vive.vive3 import VIVE3
from Submodules.r3kit.r3kit.devices.encoder.pdcd.angler import Angler
from Submodules.r3kit.r3kit.utils.transformation import xyzquat2mat

from Assets.GenieG2.joint_name_index import JOINT_NAME_TO_INDEX
from Utils.shm import shm_create, shm_write
from Utils.common import get_state, init_clock
from Utils.transform import quat_project_to_xz_plane
from Utils.solver import solver


class Args(Tap):
    tracker_id: str = 'waist'
    tracker_name: str = 'VIVE3_WAIST'

    encoder_id: str = 'COM14'
    encoder_index: List[int] = [1,2,3,4,5,6,7,9,10,11,12,13,14,15]
    encoder_baudrate: int = 1000000
    encoder_gap: float = -1
    encoder_strict: bool = True
    encoder_name: str = 'Angler'

    urdf_path: str = os.path.join(ROOT_DIR, "Assets", "GenieG2", "G2_t1_crs_omnipicker.urdf")

def main(args: Args):
    tracker = VIVE3(id=[args.tracker_id], name=args.tracker_name)
    encoder = Angler(id=args.encoder_id, index=args.encoder_index, baudrate=args.encoder_baudrate, gap=args.encoder_gap, strict=args.encoder_strict, name=args.encoder_name)

    model = pin.buildModelFromUrdf(args.urdf_path)
    data = model.createData()

    shm = shm_create()

    init_clock()
    t_prev = time.perf_counter()
    q_prev = None

    data_init = tracker.get()
    xyz_init = np.asarray(data_init["xyz"]).reshape(-1, 3)[0]
    quat_init = np.asarray(data_init["quat"]).reshape(-1, 4)[0]
    T0 = xyzquat2mat(xyz_init, quat_init)
    R_x90 = Rotation.from_euler("xyz", [np.pi / 2, -np.pi / 2, 0.0]).as_matrix()
    R_mat = np.eye(4)
    R_mat[:3, :3] = R_x90
    T0 = T0 @ R_mat
    print("Init done")

    step = 0
    warning_step = 100
    joints = np.array([0.0] * 46)
    joints[JOINT_NAME_TO_INDEX["idx01_body_joint1"]] = 0.35
    joints[JOINT_NAME_TO_INDEX["idx02_body_joint2"]] = -0.83
    joints[JOINT_NAME_TO_INDEX["idx04_body_joint4"]] = -0.4
    
    # get init body pose
    q = pin.neutral(model).copy()
    body_joint_names = ["idx01_body_joint1", "idx02_body_joint2", "idx04_body_joint4"]
    for name in body_joint_names:
        jid = model.getJointId(name)
        idx_q = model.joints[jid].idx_q
        q[idx_q] = joints[JOINT_NAME_TO_INDEX[name]]

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    torso_fid = model.getFrameId("body_link5")
    oMf = data.oMf[torso_fid]
    p_torso = oMf.translation
    init_xyz = np.array(p_torso).ravel()

    try:
        while True:
            if step <= warning_step:
                shm_write(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), joints, shm)
                print(f"[{step}] Warning")
                step += 1
                time.sleep(0.001)
                continue

            state = get_state(tracker, T0, encoder)

            joints[JOINT_NAME_TO_INDEX["idx21_arm_l_joint1"]] = state["left_arm"][0]
            joints[JOINT_NAME_TO_INDEX["idx22_arm_l_joint2"]] = state["left_arm"][1]
            joints[JOINT_NAME_TO_INDEX["idx23_arm_l_joint3"]] = state["left_arm"][2]
            joints[JOINT_NAME_TO_INDEX["idx24_arm_l_joint4"]] = state["left_arm"][3]
            joints[JOINT_NAME_TO_INDEX["idx25_arm_l_joint5"]] = state["left_arm"][4]
            joints[JOINT_NAME_TO_INDEX["idx26_arm_l_joint6"]] = state["left_arm"][5]
            joints[JOINT_NAME_TO_INDEX["idx27_arm_l_joint7"]] = state["left_arm"][6]
            joints[JOINT_NAME_TO_INDEX["idx61_arm_r_joint1"]] = state["right_arm"][0]
            joints[JOINT_NAME_TO_INDEX["idx62_arm_r_joint2"]] = state["right_arm"][1]
            joints[JOINT_NAME_TO_INDEX["idx63_arm_r_joint3"]] = state["right_arm"][2]
            joints[JOINT_NAME_TO_INDEX["idx64_arm_r_joint4"]] = state["right_arm"][3]
            joints[JOINT_NAME_TO_INDEX["idx65_arm_r_joint5"]] = state["right_arm"][4]
            joints[JOINT_NAME_TO_INDEX["idx66_arm_r_joint6"]] = state["right_arm"][5]
            joints[JOINT_NAME_TO_INDEX["idx67_arm_r_joint7"]] = state["right_arm"][6]

            quat_proj_wxyz = quat_project_to_xz_plane(state["pose_quat_wxyz"])
            xyz = state["pose_xyz"]
            xyz += init_xyz

            joints_solved = solver(
                model=model,
                data=data,
                target_xyz = xyz,
                target_quat_wxyz = quat_proj_wxyz,
                current_joints = joints,
                free_joint_names = ["idx01_body_joint1", "idx02_body_joint2", "idx04_body_joint4"],
                frame_name = "body_link5",
            )



            shm_write(state["pose_xyz"], state["pose_quat_wxyz"], joints_solved, shm=shm)

            print(f"[{step}] Written")

            step += 1
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
