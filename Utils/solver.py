import pinocchio as pin
import numpy as np
from typing import List
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from Assets.GenieG2.joint_name_index import JOINT_NAME_TO_INDEX


def solver(
    model: pin.Model,
    data: pin.Data,
    target_xyz: np.ndarray,
    target_quat_wxyz: np.ndarray,
    current_joints: np.ndarray,
    free_joint_names: List[str],
    frame_name: str,
) -> np.ndarray:
    q0 = pin.neutral(model).copy()
    free_idx_q = []
    for name in free_joint_names:
        i_q = model.joints[model.getJointId(name)].idx_q
        q0[i_q] = current_joints[JOINT_NAME_TO_INDEX[name]]
        free_idx_q.append(i_q)
    free_idx_q = np.array(free_idx_q, dtype=int)

    def cost(x):
        qc = q0.copy()
        qc[free_idx_q] = x
        
        # forward kinematics
        pin.forwardKinematics(model, data, qc)
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name)
        oMf = data.oMf[fid]
        p = np.asarray(oMf.translation)
        R = np.asarray(oMf.rotation)

        error_xyz = np.linalg.norm(p - target_xyz)**2
        q_xyzw = [target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]]
        R_target = Rotation.from_quat(q_xyzw).as_matrix()
        error_R = R_target.T @ R
        theta = np.arccos(np.clip((np.trace(error_R) - 1)/2, -1.0, 1.0))
        error_rot = theta**2

        return error_xyz + error_rot

    x0 = q0[free_idx_q]
    bounds = [(model.lowerPositionLimit[i], model.upperPositionLimit[i]) for i in free_idx_q]
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds)
    q_solved = q0.copy()
    q_solved[free_idx_q] = res.x
    
    joints_solved = np.array(current_joints, copy=True)
    for name in free_joint_names:
        i_q = model.joints[model.getJointId(name)].idx_q
        joints_solved[JOINT_NAME_TO_INDEX[name]] = float(q_solved[i_q])

    return joints_solved