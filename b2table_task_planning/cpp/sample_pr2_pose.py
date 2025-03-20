import ctypes
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm

global_ineq = []

INEQ_CST_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_double))
this_dir_path = Path(__file__).parent
so_file_path = (this_dir_path / "_sample_pr2_pose.so").resolve()
lib = ctypes.cdll.LoadLibrary(str(so_file_path))

lib.create_sampler.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # region_size
    ctypes.POINTER(ctypes.c_double),  # region_center
    ctypes.POINTER(ctypes.c_double),  # table_size
    ctypes.POINTER(ctypes.c_double),  # table_center
    INEQ_CST_FUNC_TYPE,  # ineq_cst_pred
    ctypes.c_int,  # seed
]
lib.create_sampler.restype = ctypes.c_void_p

lib.sample_pose.argtypes = [
    ctypes.c_void_p,  # sampler (void*)
    ctypes.POINTER(ctypes.c_double),  # reaching_pose
    ctypes.POINTER(ctypes.c_double),  # pose_out
]
lib.sample_pose.restype = ctypes.c_bool

lib.destroy_sampler.argtypes = [ctypes.c_void_p]
lib.destroy_sampler.restype = None


def create_sampler(
    region_size: np.ndarray,
    region_center: np.ndarray,
    table_size: np.ndarray,
    table_center: np.ndarray,
    ineq_cst_pred,
    seed: int,
) -> ctypes.c_void_p:
    region_size = region_size.astype(np.double)
    region_center = region_center.astype(np.double)
    table_size = table_size.astype(np.double)
    ineq_cst_pred = INEQ_CST_FUNC_TYPE(ineq_cst_pred)

    global_ineq.append(ineq_cst_pred)

    return lib.create_sampler(
        region_size.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        region_center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        table_size.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        table_center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ineq_cst_pred,
        seed,
    )


def sample_pose(sampler: ctypes.c_void_p, reaching_pose: np.ndarray) -> Optional[np.ndarray]:
    reaching_pose = reaching_pose.astype(np.double)
    pose_out = np.zeros(3)
    pose_out = pose_out.astype(np.double)
    success = lib.sample_pose(
        sampler,
        reaching_pose.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pose_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if success:
        return pose_out
    else:
        return None


def destroy_sampler(sampler: ctypes.c_void_p) -> None:
    lib.destroy_sampler(sampler)


if __name__ == "__main__":
    region_size = np.array([2.0, 2.0])
    region_center = np.array([0.0, 0.0])
    table_size = np.array([1.0, 1.0])
    table_center = np.array([0.0, 0.0])

    def ineq_cst_pred(x) -> bool:
        return 1

    sampler = create_sampler(region_size, region_center, table_size, table_center, ineq_cst_pred, 0)
    out = np.zeros(3)

    for _ in tqdm.tqdm(range(1000000)):
        pose = sample_pose(sampler, np.array([0.0, 0.0, 0.0, 0.0]), out)
