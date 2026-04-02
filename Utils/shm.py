import struct
from multiprocessing import shared_memory

SHM_NAME = "shm_agibotsim"
NUM_JOINTS = 46
POSE_FLOATS = 7
SHM_FLOATS = POSE_FLOATS + NUM_JOINTS
SHM_BYTES = SHM_FLOATS * 8


def shm_create() -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_BYTES)
    except FileExistsError:
        return shared_memory.SharedMemory(name=SHM_NAME, create=False)


def shm_write(pose_xyz, pose_quat_wxyz, joint_positions, shm: shared_memory.SharedMemory | None = None) -> None:
    flat = tuple(float(x) for x in pose_xyz) + tuple(float(q) for q in pose_quat_wxyz) + tuple(float(j) for j in joint_positions)
    if shm is not None:
        struct.pack_into(f"{SHM_FLOATS}d", shm.buf, 0, *flat)
        return
    s = shared_memory.SharedMemory(name=SHM_NAME, create=False)
    try:
        struct.pack_into(f"{SHM_FLOATS}d", s.buf, 0, *flat)
    finally:
        s.close()


def shm_read():
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
    except FileNotFoundError:
        return None
    try:
        if len(shm.buf) < SHM_BYTES:
            return None
        raw = struct.unpack_from(f"{SHM_FLOATS}d", shm.buf, 0)
        return (raw[0:3], raw[3:7], raw[7:7 + NUM_JOINTS])
    finally:
        shm.close()
