import time
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation
from Submodules.r3kit.r3kit.devices.tracker.vive.vive3 import VIVE3
from Submodules.r3kit.r3kit.devices.encoder.pdcd.angler import Angler
from Submodules.r3kit.r3kit.utils.transformation import xyzquat2mat, mat2xyzquat

_t0: Optional[float] = None


def init_clock() -> None:
    global _t0
    _t0 = time.perf_counter()


def get_state(tracker: VIVE3, T0: np.ndarray, encoder: Angler) -> dict:
    tracker_data = tracker.get()
    xyz = np.asarray(tracker_data["xyz"]).reshape(-1, 3)[0]
    quat = np.asarray(tracker_data["quat"]).reshape(-1, 4)[0]
    T = xyzquat2mat(xyz, quat)
    R_x90 = Rotation.from_euler("xyz", [np.pi / 2, -np.pi / 2, 0.0]).as_matrix()
    R_mat = np.eye(4)
    R_mat[:3, :3] = R_x90
    T = T @ R_mat
    delta_T = np.linalg.inv(T0) @ T
    delta_xyz, delta_quat = mat2xyzquat(delta_T)  # [x, y, z, w]
    delta_quat_wxyz = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])
    delta_quat_xyzw = np.array([delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3]])
    delta_rpy = Rotation.from_quat(delta_quat_xyzw).as_euler("xyz")

    encoder_data = encoder.get()
    left_arm = mapping_left(encoder_data["angle"][:7])
    right_arm = mapping_right(encoder_data["angle"][7:])

    # return {
    #     "t": tracker_data["timestamp_ms"],
    #     "pose_xyz": delta_xyz,
    #     "pose_rpy": delta_rpy,
    #     "pose_quat_wxyz": delta_quat_wxyz,
    #     "left_arm": left_arm,
    #     "right_arm": right_arm,
    # }
    return {
        "t": tracker_data["timestamp_ms"],
        "pose_xyz": delta_xyz,
        "pose_rpy": delta_rpy,
        "pose_quat_wxyz": delta_quat_wxyz,
        "left_arm": left_arm,
        "right_arm": right_arm,
    }


def mapping_left(
    angles: np.ndarray,
    middles: np.ndarray = np.array([246.62, 140.54, 280.99, 224.82, 256.64, 84.64, 308.23]),
) -> np.ndarray:
    angles = np.asarray(angles, dtype=float).copy()

    # First 7 channels: arm joints (degrees -> centered degrees -> radians)
    joints_deg = (angles[:7] - middles[:7] + 180.0) % 360.0 - 180.0
    joints_deg[0] = -joints_deg[0]
    joints_deg[1] = -joints_deg[1]
    joints_deg[2] = -joints_deg[2]
    joints_deg[3] = -joints_deg[3]
    joints_deg[4] = -joints_deg[4]
    joints_deg[5] = joints_deg[5]
    joints_deg[6] = -joints_deg[6]

    return np.deg2rad(joints_deg)


def mapping_right(
    angles: np.ndarray,
    middles: np.ndarray = np.array([236.78, 298.21, 175.52, 180.18, 189.23, 347.34, 154.25]),
) -> np.ndarray:
    angles = np.asarray(angles, dtype=float).copy()

    # First 7 channels: arm joints (degrees -> centered degrees -> radians)
    joints_deg = (angles[:7] - middles[:7] + 180.0) % 360.0 - 180.0
    joints_deg[0] = -joints_deg[0]
    joints_deg[1] = joints_deg[1]
    joints_deg[2] = -joints_deg[2]
    joints_deg[3] = joints_deg[3]
    joints_deg[4] = -joints_deg[4]
    joints_deg[5] = -joints_deg[5]
    joints_deg[6] = -joints_deg[6]

    return np.deg2rad(joints_deg)
