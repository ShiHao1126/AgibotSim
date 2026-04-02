import math
import numpy as np
from scipy.spatial.transform import Rotation


def quat_project_to_xy_plane(q_wxyz: np.ndarray) -> np.ndarray:
    """Project quaternion to only rotate around world z-axis, input/output (w, x, y, z)."""
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    r = Rotation.from_quat(q_xyzw)
    R = r.as_matrix()
    heading = math.atan2(R[1, 0], R[0, 0])
    r_proj = Rotation.from_euler("z", heading)
    q_proj_xyzw = r_proj.as_quat()
    return np.array(
        [q_proj_xyzw[3], q_proj_xyzw[0], q_proj_xyzw[1], q_proj_xyzw[2]],
        dtype=np.float64,
    )


def quat_project_to_xz_plane(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Extract pitch (tilt in world xz-plane) and return a quaternion that only rotates around
    world y-axis. Input/output (w, x, y, z).

    The extracted pitch angle is clamped to ±30 degrees.
    """
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    r = Rotation.from_quat(q_xyzw)
    R = r.as_matrix()

    # Use body z-axis direction in world frame (third column of R) to extract pitch in xz-plane.
    # For Ry(theta): body z-axis becomes [sin(theta), 0, cos(theta)] => theta = atan2(zx, zz)
    theta = math.atan2(R[0, 2], R[2, 2])

    limit = math.radians(30.0)
    theta = max(-limit, min(limit, theta))

    r_proj = Rotation.from_euler("y", theta)
    q_proj_xyzw = r_proj.as_quat()
    return np.array(
        [q_proj_xyzw[3], q_proj_xyzw[0], q_proj_xyzw[1], q_proj_xyzw[2]],
        dtype=np.float64,
    )