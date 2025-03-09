import time

import numpy as np
import torch
import tqdm
from hifuku.core import SolutionLibrary
from hifuku.domain import Pr2ThesisJskTable, Pr2ThesisJskTable2
from hifuku.script_utils import load_library
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PR2BaseOnlySpec, PR2LarmSpec, PR2RarmSpec
from rpbench.articulated.pr2.thesis_jsk_table import (
    AV_INIT,
    JskChair,
    JskMessyTableTask,
    JskMessyTableTaskWithChair,
    JskTable,
)
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.utils import BoxSkeleton
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from b2table_task_planning.sampler import SituationSampler
from b2table_task_planning.scenario import barely_feasible_task


def create_map_from_obstacle_param(obstacles_param: np.ndarray):
    obstacle_env_region = BoxSkeleton(
        [
            JskTable.TABLE_DEPTH,
            JskTable.TABLE_WIDTH,
            JskMessyTableTask.OBSTACLE_H_MAX + 0.05,
        ]
    )
    obstacle_env_region.translate(
        [0.0, 0.0, JskTable.TABLE_HEIGHT + obstacle_env_region.extents[2] * 0.5]
    )

    obstacles = []
    for obs_param in obstacles_param.reshape(-1, 6):
        x, y, yaw, d, w, h = obs_param
        box = BoxSkeleton([d, w, h], pos=[x, y, JskTable.TABLE_HEIGHT + 0.5 * h])
        box.rotate(yaw, "z")
        obstacles.append(box)
    table_mat = create_heightmap_z_slice(obstacle_env_region, obstacles, 112)
    return table_mat


def create_map_from_chair_param(chairs_param: np.ndarray):
    ground_region, _ = JskMessyTableTaskWithChair._prepare_target_region()
    primitives = []
    for chair_param in chairs_param.reshape(-1, 3):
        x, y, yaw = chair_param
        chair = JskChair()
        chair.translate([x, y, 0.0])
        chair.rotate(yaw, "z")
        primitives.extend(chair.chair_primitives)
    ground_mat = create_heightmap_z_slice(ground_region, primitives, 112)
    return ground_mat


class _FeasibilityChecker:
    lib: SolutionLibrary

    def __init__(self):
        self.lib = load_library(Pr2ThesisJskTable, "cuda", postfix="0.2")
        # Not jit compiled intentionally becaues input shape is not fixed
        task = JskMessyTableTask.sample()
        for _ in range(10):  # warm up
            self.lib.infer(task)

    def infer(self, vectors: np.ndarray, table_mat: np.ndarray):
        assert self.lib.ae_model_shared is not None
        vectors_cuda = torch.from_numpy(vectors).float().cuda()
        table_mat_cuda = torch.from_numpy(table_mat).float().cuda().unsqueeze(0).unsqueeze(0)
        encoded = self.lib.ae_model_shared.forward(table_mat_cuda)
        n_batch = vectors_cuda.shape[0]
        encoded_repeated = encoded.repeat(n_batch, 1)

        costs_arr = torch.zeros(len(self.lib.predictors), n_batch).float().cuda()
        for i in range(len(self.lib.predictors)):
            pred = self.lib.predictors[i]
            bias = self.lib.biases[i]
            costs = pred.forward((encoded_repeated, vectors_cuda))[0]
            costs_arr[i] = costs.flatten() + bias
        min_costs, min_indices = torch.min(costs_arr, dim=0)
        return (
            min_costs.cpu().detach().numpy() < self.lib.max_admissible_cost,
            min_indices.cpu().detach().numpy(),
        )


class ChairManager:
    def __init__(self):
        n_max_chair = JskMessyTableTaskWithChair.N_MAX_CHAIR
        self.chairs = [JskChair() for _ in range(n_max_chair)]
        self.sdfs = [None for _ in range(n_max_chair)]
        self.n_chair = 0

    def set_param(self, chairs_param: np.ndarray):
        reshaped = chairs_param.reshape(-1, 3)
        n_chair = reshaped.shape[0]
        self.n_chair = n_chair
        for i in range(n_chair):
            x, y, yaw = reshaped[i]
            self.chairs[i].newcoords(Coordinates([x, y, 0.0], [yaw, 0.0, 0.0]))
            print(x, y, yaw)
            self.sdfs[i] = self.chairs[i].create_sdf()

    def create_sdf(self) -> UnionSDF:
        sdf = UnionSDF([])
        for i in range(self.n_chair):
            sdf.merge(self.sdfs[i])
        return sdf


class FeasibilityChecker:
    lib: SolutionLibrary

    def __init__(self):
        self.lib = load_library(Pr2ThesisJskTable2, "cuda", postfix="0.2")
        # Not jit compiled intentionally becaues input shape is not fixed
        task = JskMessyTableTaskWithChair.sample()
        for _ in range(10):  # warm up
            self.lib.infer(task)

    def infer(self, vectors: np.ndarray, table_mat: np.ndarray, ground_mat: np.ndarray):
        vectors = torch.from_numpy(vectors).float().cuda()
        table_mat = torch.from_numpy(table_mat).float().cuda()  # 112 x 112
        ground_mat = torch.from_numpy(ground_mat).float().cuda()  # 112 x 112
        # create (2 x 112 x 112) tensor
        mat = torch.stack([table_mat, ground_mat], dim=0).unsqueeze(0)

        encoded = self.lib.ae_model_shared.forward(mat)
        n_batch = vectors.shape[0]
        encoded_repeated = encoded.repeat(n_batch, 1)

        costs_arr = torch.zeros(len(self.lib.predictors), n_batch).float().cuda()
        for i in range(len(self.lib.predictors)):
            pred = self.lib.predictors[i]
            bias = self.lib.biases[i]
            costs = pred.forward((encoded_repeated, vectors))[0]
            costs_arr[i] = costs.flatten() + bias
        min_costs, min_indices = torch.min(costs_arr, dim=0)
        return (
            min_costs.cpu().detach().numpy() < self.lib.max_admissible_cost,
            min_indices.cpu().detach().numpy(),
        )


if __name__ == "__main__":
    sampler = SituationSampler()
    task = barely_feasible_task()
    chair_manager = ChairManager()

    engine = FeasibilityChecker()

    sampler.register_tabletop_obstacles(task.obstacles_param)
    sampler.register_reaching_pose(task.reaching_pose)
    table_mat = create_map_from_obstacle_param(task.obstacles_param)
    ground_mat = create_map_from_chair_param(task.chairs_param)

    pose_list = []
    for _ in tqdm.tqdm(range(1000)):
        pose = sampler.sample_pr2_pose()
        if pose is not None:
            pose_list.append(pose)
    pose_list = np.array(pose_list)

    chair_manager.set_param(task.chairs_param)
    sdf = chair_manager.create_sdf()

    pr2_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
    pr2_spec.get_kin()
    skmodel = pr2_spec.get_robot_model()
    skmodel.angle_vector(AV_INIT)
    pr2_spec.reflect_skrobot_model_to_kin(
        skmodel
    )  # NOTE: robot configuration expect for base is fixed
    cst = pr2_spec.create_collision_const()
    cst.set_sdf(sdf)

    valid_pose_list = []
    for pose in pose_list:
        valid = cst.is_valid(pose)
        if valid:
            valid_pose_list.append(pose)
    valid_pose_list = np.array(valid_pose_list)

    reaching_pose_tile = np.tile(task.reaching_pose, (valid_pose_list.shape[0], 1))
    vectors = np.concatenate([valid_pose_list, reaching_pose_tile], axis=1)

    is_feasibiles, min_indices = engine.infer(vectors, table_mat, ground_mat)

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.add_patch(
            plt.Rectangle(
                (-0.5 * JskTable.TABLE_DEPTH, -0.5 * JskTable.TABLE_WIDTH),
                JskTable.TABLE_DEPTH,
                JskTable.TABLE_WIDTH,
                fill=False,
            )
        )

        for (x, y, yaww) in pose_list:
            color = "gray"
            dx = np.cos(yaww) * 0.03
            dy = np.sin(yaww) * 0.03
            ax.arrow(x, y, dx, dy, head_width=0.01, length_includes_head=True, color=color)

        for (x, y, yaw), is_feasible in zip(valid_pose_list, is_feasibiles):
            color = "blue" if is_feasible else "red"
            dx = np.cos(yaw) * 0.03
            dy = np.sin(yaw) * 0.03
            ax.arrow(x, y, dx, dy, head_width=0.01, length_includes_head=True, color=color)

        plt.axis("equal")
        plt.show()
    else:
        domain = Pr2ThesisJskTable
        solver = domain.solver_type.init(domain.solver_config)
        for pose, is_feasible, min_idx in zip(valid_pose_list, is_feasibiles, min_indices):
            if is_feasible:
                task.pr2_coords = pose
                solver.setup(task.export_problem())
                init_traj = engine.lib.init_solutions[min_idx]
                res = solver.solve(init_traj)
                if task.is_using_rarm():
                    spec = PR2RarmSpec()
                else:
                    spec = PR2LarmSpec()
                v = PyrenderViewer()
                task.visualize(v)
                pr2 = PR2(use_tight_joint_limit=False)
                pr2.angle_vector(AV_INIT)
                pr2.newcoords(Coordinates([pose[0], pose[1], 0.0], [pose[2], 0.0, 0.0]))
                v.add(pr2)
                v.show()
                time.sleep(2)
                for q in res.traj:
                    spec.set_skrobot_model_state(pr2, q)
                    v.redraw()
                    time.sleep(0.05)
                import time

                time.sleep(1000)
