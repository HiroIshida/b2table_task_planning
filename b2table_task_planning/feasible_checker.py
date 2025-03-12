import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tqdm
from hifuku.core import SolutionLibrary
from hifuku.domain import Pr2ThesisJskTable, Pr2ThesisJskTable2
from hifuku.script_utils import load_library
from plainmp.experimental import MultiGoalRRT
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PR2BaseOnlySpec, PR2LarmSpec, PR2RarmSpec
from plainmp.trajectory import Trajectory
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


@dataclass
class TaskPlannerConfig:
    n_sample_pose: int = 1000


@dataclass
class TaskAndMotionPlan:
    pr2_pose: np.ndarray
    traj: Trajectory
    rarm: bool


class TaskPlanner:
    sampler: SituationSampler
    chair_manager: ChairManager
    engine: FeasibilityChecker
    config: TaskPlannerConfig

    def __init__(self, config: TaskPlannerConfig = TaskPlannerConfig()):
        self.sampler = SituationSampler()
        self.chair_manager = ChairManager()
        self.engine = FeasibilityChecker()
        self.config = config
        self.table = JskTable()

        target_region, _ = JskMessyTableTaskWithChair._prepare_target_region()
        region_lb = target_region.worldpos() - 0.5 * target_region.extents
        region_ub = target_region.worldpos() + 0.5 * target_region.extents
        self.pr2_pose_lb = np.hstack([region_lb[:2], [-np.pi * 1.5]])
        self.pr2_pose_ub = np.hstack([region_ub[:2], [+np.pi * 1.5]])

    def plan(
        self,
        pr2_pose_now: np.ndarray,
        reaching_pose: np.ndarray,
        obstacles_param: np.ndarray,
        chairs_param: np.ndarray,
    ) -> Optional[TaskAndMotionPlan]:

        # here we assume that only chair is movable
        self.sampler.register_tabletop_obstacles(obstacles_param)
        self.sampler.register_reaching_pose(reaching_pose)
        self.chair_manager.set_param(chairs_param)
        pr2_pose_cands = self._sample_pr2_pose()
        if pr2_pose_cands is None:
            return None  # no solution found
        placable_pr2_poses = self._select_reachable_poses(pr2_pose_now, pr2_pose_cands)

        if placable_pr2_poses is None:
            raise NotImplementedError("No placable pose found")  # TODO: handle this case

        # finally
        table_mat = create_map_from_obstacle_param(obstacles_param)
        ground_mat = create_map_from_chair_param(chairs_param)

        reaching_pose_tile = np.tile(reaching_pose, (placable_pr2_poses.shape[0], 1))
        vectors = np.concatenate([placable_pr2_poses, reaching_pose_tile], axis=1)
        is_feasibiles, min_indices = self.engine.infer(vectors, table_mat, ground_mat)

        # check if any feasible solution exists
        if not np.any(is_feasibiles):
            return NotImplementedError("No feasible solution found")

        print("now the phase of finding feasible solution by actually solving the problem")
        for pose, is_feasible, min_idx in zip(placable_pr2_poses, is_feasibiles, min_indices):
            if is_feasible:
                task = JskMessyTableTaskWithChair(
                    obstacles_param, chairs_param, pose, reaching_pose
                )
                solver = Pr2ThesisJskTable2.solver_type.init(Pr2ThesisJskTable2.solver_config)
                solver.setup(task.export_problem())
                init_traj = self.engine.lib.init_solutions[min_idx]
                res = solver.solve(init_traj)
                if res.traj is not None:
                    return TaskAndMotionPlan(
                        pr2_pose=pose, traj=res.traj, rarm=task.is_using_rarm()
                    )

    def _sample_pr2_pose(self) -> Optional[np.ndarray]:
        pose_list = []
        for _ in tqdm.tqdm(range(self.config.n_sample_pose)):
            pose = self.sampler.sample_pr2_pose()
            if pose is not None:
                pose_list.append(pose)
        if len(pose_list) == 0:
            return None  # no solution found
        pose_list = np.array(pose_list)
        return pose_list

    def _select_reachable_poses(
        self, current_pose: np.ndarray, pose_list: np.ndarray
    ) -> Optional[np.ndarray]:
        # check if robot is placable at the sampled pose
        sdf = self.chair_manager.create_sdf()
        sdf.merge(self.table.create_sdf())

        pr2_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        pr2_spec.get_kin()
        skmodel = pr2_spec.get_robot_model()
        skmodel.angle_vector(AV_INIT)
        pr2_spec.reflect_skrobot_model_to_kin(skmodel)
        cst = pr2_spec.create_collision_const()
        cst.set_sdf(sdf)

        tree = MultiGoalRRT(current_pose, self.pr2_pose_lb, self.pr2_pose_ub, cst, 1000)
        bools = tree.is_reachable_batch(pose_list.T, 0.5)
        return pose_list[bools]

    def _determine_remove_chair(self, pose: np.ndarray):
        pass


if __name__ == "__main__":
    task = barely_feasible_task()
    task_planner = TaskPlanner()

    start = np.array([0.784, 2.57, -2.0])
    ts = time.time()
    plan = task_planner.plan(start, task.reaching_pose, task.obstacles_param, task.chairs_param)
    print(f"Time: {time.time() - ts}")

    if plan is not None:
        spec = PR2RarmSpec() if plan.rarm else PR2LarmSpec()
        v = PyrenderViewer()
        task.visualize(v)
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.angle_vector(AV_INIT)
        pr2.newcoords(
            Coordinates([plan.pr2_pose[0], plan.pr2_pose[1], 0.0], [plan.pr2_pose[2], 0.0, 0.0])
        )
        v.add(pr2)
        v.show()
        time.sleep(2)
        for q in plan.traj:
            spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.05)
        time.sleep(1000)
