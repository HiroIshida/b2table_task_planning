import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tqdm
from hifuku.core import SolutionLibrary
from hifuku.domain import Pr2ThesisJskTable2
from hifuku.script_utils import load_library
from plainmp.constraint import SphereCollisionCst
from plainmp.experimental import MultiGoalRRT
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import (
    PR2BaseOnlySpec,
    PR2LarmSpec,
    PR2RarmSpec,
    SphereAttachmentSpec,
)
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
from b2table_task_planning.scenario import need_fix_task


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


class ChairManager:
    def __init__(self):
        n_max_chair = JskMessyTableTaskWithChair.N_MAX_CHAIR
        self.chairs_param = None
        self.chairs = [JskChair() for _ in range(n_max_chair)]
        self.sdfs = [None for _ in range(n_max_chair)]
        self.n_chair = 0

    def set_param(self, chairs_param: np.ndarray):
        self.chairs_param = chairs_param
        reshaped = chairs_param.reshape(-1, 3)
        n_chair = reshaped.shape[0]
        self.n_chair = n_chair
        for i in range(n_chair):
            x, y, yaw = reshaped[i]
            self.chairs[i].newcoords(Coordinates([x, y, 0.0], [yaw, 0.0, 0.0]))
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


@dataclass
class PlanningResult:
    joint_path_final: Optional[Trajectory] = None
    is_rarm: Optional[bool] = None
    base_path_final: Optional[Trajectory] = None
    base_path_to_post_remove_chair: Optional[Trajectory] = None
    base_path_to_pre_remove_chair: Optional[Trajectory] = None


class TaskPlanner:
    sampler: SituationSampler
    chair_manager: ChairManager
    engine: FeasibilityChecker
    config: TaskPlannerConfig
    table: JskTable
    pr2_pose_lb: np.ndarray
    pr2_pose_ub: np.ndarray

    def __init__(self, config: TaskPlannerConfig = TaskPlannerConfig()):
        self.sampler = SituationSampler()
        self.chair_manager = ChairManager()
        self.engine = FeasibilityChecker()
        self.config = config
        self.table = JskTable()

        target_region, _ = JskMessyTableTaskWithChair._prepare_target_region()
        region_lb = target_region.worldpos() - 0.5 * target_region.extents
        region_ub = target_region.worldpos() + 0.5 * target_region.extents
        region_ub[1] += 0.6
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
        recahable_pr2_poses = self._select_reachable_poses(pr2_pose_now, pr2_pose_cands)

        if recahable_pr2_poses is None:
            rplanner = RepairPlanner(
                pr2_pose_now,
                pr2_pose_cands,
                reaching_pose,
                self.chair_manager,
                self.engine,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                obstacles_param,
                self.table,
                create_map_from_obstacle_param(obstacles_param),
            )
            return rplanner.plan(chairs_param)

        # finally
        table_mat = create_map_from_obstacle_param(obstacles_param)
        ground_mat = create_map_from_chair_param(chairs_param)

        reaching_pose_tile = np.tile(reaching_pose, (recahable_pr2_poses.shape[0], 1))
        vectors = np.concatenate([recahable_pr2_poses, reaching_pose_tile], axis=1)
        is_feasibiles, min_indices = self.engine.infer(vectors, table_mat, ground_mat)

        # check if any feasible solution exists
        if not np.any(is_feasibiles):
            rplanner = RepairPlanner(
                pr2_pose_now,
                pr2_pose_cands,
                reaching_pose,
                self.chair_manager,
                self.engine,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                obstacles_param,
                self.table,
                table_mat,
            )
            return rplanner.plan(chairs_param)

        print("now the phase of finding feasible solution by actually solving the problem")
        for pose, is_feasible, min_idx in zip(recahable_pr2_poses, is_feasibiles, min_indices):
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
        skmodel = pr2_spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        pr2_spec.reflect_skrobot_model_to_kin(skmodel)
        cst = pr2_spec.create_collision_const()
        cst.set_sdf(sdf)

        tree = MultiGoalRRT(current_pose, self.pr2_pose_lb, self.pr2_pose_ub, cst, 1000)
        bools = tree.is_reachable_batch(pose_list.T, 0.5)
        return pose_list[bools]


class RepairPlanner:
    init_pr2_pose: np.ndarray
    final_pr2_pose_cands: np.ndarray
    final_gripper_pose: np.ndarray
    chair_manager: ChairManager
    engine: FeasibilityChecker
    pr2_pose_lb: np.ndarray
    pr2_pose_ub: np.ndarray
    obstacle_param: np.ndarray
    table: JskTable
    table_mat: np.ndarray  # heightmap of the table (can be genrated from obstacle_param but we cache it)
    collision_cst_base_only: SphereCollisionCst
    collision_cst_with_chair: SphereCollisionCst
    CHAIR_GRASP_BASE_OFFSET = 0.8

    def __init__(
        self,
        init_pr2_pose: np.ndarray,
        final_pr2_pose_cands: np.ndarray,
        final_gripper_pose: np.ndarray,
        chair_manager: ChairManager,
        engine: FeasibilityChecker,
        pr2_pose_lb: np.ndarray,
        pr2_pose_ub: np.ndarray,
        obstacle_param: np.ndarray,
        table: JskTable,
        table_mat: np.ndarray,
    ):
        self.init_pr2_pose = init_pr2_pose
        self.final_pr2_pose_cands = final_pr2_pose_cands
        self.final_gripper_pose = final_gripper_pose
        self.chair_manager = chair_manager
        self.engine = engine
        self.pr2_pose_lb = pr2_pose_lb
        self.pr2_pose_ub = pr2_pose_ub
        self.obstacle_param = obstacle_param
        self.table = table
        self.table_mat = table_mat

        spec = PR2BaseOnlySpec(use_fixed_uuid=True)

        # prepare collision constraint
        skmodel = spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        spec.reflect_skrobot_model_to_kin(skmodel)
        self.collision_cst_base_only = spec.create_collision_const()

        # prepare chair-attached collision constraint
        chair_bbox_lb = np.array([-JskChair.DEPTH * 0.5, -JskChair.WIDTH * 0.5, 0.0])
        chair_bbox_ub = np.array([JskChair.DEPTH * 0.5, JskChair.WIDTH * 0.5, JskChair.BACK_HEIGHT])
        x = np.linspace(chair_bbox_lb[0], chair_bbox_ub[0], 5) + self.CHAIR_GRASP_BASE_OFFSET
        y = np.linspace(chair_bbox_lb[1], chair_bbox_ub[1], 5)
        z = np.linspace(chair_bbox_lb[2], chair_bbox_ub[2], 10)
        xx, yy, zz = np.meshgrid(x, y, z)
        cloud = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        aspec = SphereAttachmentSpec("base_footprint", cloud.T, np.ones(len(cloud)) * 0.03, False)
        coll_cst_with_chair = spec.create_collision_const(attachements=(aspec,))
        self.collision_cst_with_chair = coll_cst_with_chair

    def plan(self, chairs_param_original: np.ndarray) -> Optional[PlanningResult]:
        n_chair = len(chairs_param_original) // 3
        for i_chair in range(n_chair):
            chair_pose_remove = chairs_param_original[i_chair * 3 : (i_chair + 1) * 3]
            chairs_param_hypo = np.delete(
                chairs_param_original, np.s_[3 * i_chair : 3 * i_chair + 3]
            )
            self.chair_manager.set_param(chairs_param_hypo)
            sdf = self.chair_manager.create_sdf()
            sdf.merge(self.table.create_sdf())
            self.collision_cst_base_only.set_sdf(sdf)

            planning_result = PlanningResult()

            # first check if the robot base can reach the goal
            tree = MultiGoalRRT(
                self.init_pr2_pose,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                self.collision_cst_base_only,
                1000,
            )
            bools = tree.is_reachable_batch(self.final_pr2_pose_cands.T, 0.5)
            reachable_poses = self.final_pr2_pose_cands[bools]
            if len(reachable_poses) == 0:
                continue

            # check if the robot arm can reach the goal pose
            ground_mat = create_map_from_chair_param(chairs_param_hypo)
            gripper_pose_tile = np.tile(self.final_gripper_pose, (reachable_poses.shape[0], 1))
            vectors = np.concatenate([reachable_poses, gripper_pose_tile], axis=1)
            is_feasibiles, min_indices = self.engine.infer(vectors, self.table_mat, ground_mat)
            if not np.any(is_feasibiles):
                continue

            # check if i_chair can be graspable
            pre_remove_pr2_pose = self.determine_pregrasp_chair_pr2_base_pose(
                chair_pose_remove, tree
            )
            if pre_remove_pr2_pose is None:
                continue
            planning_result.base_path_to_pre_remove_chair = Trajectory(
                tree.get_solution(pre_remove_pr2_pose).T
            )

            # check if the detected situation is feasible by actually solving the problem
            feasible_pr2_final_pose = None
            for pr2_final_pose, is_feasible, min_idx in zip(
                reachable_poses, is_feasibiles, min_indices
            ):
                if not is_feasible:
                    continue
                reaching_task = JskMessyTableTaskWithChair(
                    self.obstacle_param, chairs_param_hypo, pr2_final_pose, self.final_gripper_pose
                )
                solver = Pr2ThesisJskTable2.solver_type.init(Pr2ThesisJskTable2.solver_config)
                solver.setup(reaching_task.export_problem())
                init_traj = self.engine.lib.init_solutions[min_idx]
                res = solver.solve(init_traj)
                if res.traj is None:
                    continue
                feasible_pr2_final_pose = pr2_final_pose
                planning_result.joint_path_final = res.traj
                planning_result.is_rarm = reaching_task.is_using_rarm()
                break
            if feasible_pr2_final_pose is None:
                continue

            # check if feasible placment of the chair is possible
            path = tree.get_solution(feasible_pr2_final_pose).T
            traj = Trajectory(path).resample(100)

            self.collision_cst_with_chair.set_sdf(sdf)
            tree_chair_attach = MultiGoalRRT(
                pre_remove_pr2_pose,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                self.collision_cst_with_chair,
                1000,
            )
            nodes = tree_chair_attach.get_debug_states()
            dists = np.linalg.norm(nodes[:, :2] - path[-1, :2], axis=1)
            sorted_indices = np.argsort(dists)

            valid_post_remove_pr2_pose = None
            valid_post_remove_chair_pose = None
            for ind in sorted_indices:
                post_remove_pr2_pose = nodes[ind]
                chair_pose = post_remove_pr2_pose + np.array(
                    [
                        self.CHAIR_GRASP_BASE_OFFSET * np.cos(post_remove_pr2_pose[2]),
                        self.CHAIR_GRASP_BASE_OFFSET * np.sin(post_remove_pr2_pose[2]),
                        0.0,
                    ]
                )
                self.chair_manager.set_param(chair_pose)
                sdf = self.chair_manager.create_sdf()
                self.collision_cst_base_only.set_sdf(sdf)
                is_collision_free = np.all(
                    [self.collision_cst_base_only.is_valid(q) for q in traj.numpy()]
                )
                if not is_collision_free:
                    continue
                valid_post_remove_pr2_pose = post_remove_pr2_pose
                valid_post_remove_chair_pose = chair_pose
                break
            if valid_post_remove_pr2_pose is None:
                continue
            planning_result.base_path_to_post_remove_chair = Trajectory(
                tree_chair_attach.get_solution(valid_post_remove_pr2_pose).T
            )

            # finalizing the plan connecting post_remove_pr2_pose and final_pr2_pose
            chairs_param_post_remove = np.hstack([chairs_param_hypo, valid_post_remove_chair_pose])
            self.chair_manager.set_param(chairs_param_post_remove)
            sdf_post_remove = self.chair_manager.create_sdf()
            self.collision_cst_base_only.set_sdf(sdf_post_remove)

            solver = OMPLSolver(OMPLSolverConfig(shortcut=True, bspline=True))
            problem = Problem(
                post_remove_pr2_pose,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                pr2_final_pose,
                self.collision_cst_base_only,
                None,
                np.array([0.1, 0.1, 0.1]),
            )
            ret = solver.solve(problem)
            planning_result.base_path_final = ret.traj
            return planning_result

        assert None

    def determine_pregrasp_chair_pr2_base_pose(
        self, chair_pose: np.ndarray, tree: MultiGoalRRT
    ) -> Optional[np.ndarray]:
        # check if blocking chair is actually removable by
        # hypothetically chaing the chair yaw angles
        # assuming that PR2 can rotate the chair by 90 degrees
        x, y, yaw = chair_pose
        yaw_rotates = np.linspace(-np.pi * 0.5, np.pi * 0.5, 10)
        yaw_cands = yaw + yaw_rotates
        sins = np.sin(yaw_cands)
        coss = np.cos(yaw_cands)

        xs = x - self.CHAIR_GRASP_BASE_OFFSET * coss
        ys = y - self.CHAIR_GRASP_BASE_OFFSET * sins
        pr2_pose_pre_grasp_cands = np.array([xs, ys, yaw_cands]).T
        bools = tree.is_reachable_batch(pr2_pose_pre_grasp_cands.T, 0.5)
        if not np.any(bools):
            return None

        min_yaw = yaw_cands[bools].min()
        x = x - self.CHAIR_GRASP_BASE_OFFSET * np.cos(min_yaw)
        y = y - self.CHAIR_GRASP_BASE_OFFSET * np.sin(min_yaw)
        pre_grasp_base_pose = np.array([x, y, min_yaw])
        return pre_grasp_base_pose


if __name__ == "__main__":
    # task = barely_feasible_task()
    task = need_fix_task()
    task_planner = TaskPlanner()

    start = np.array([0.784, 2.57, -2.0])
    ts = time.time()
    plan = task_planner.plan(start, task.reaching_pose, task.obstacles_param, task.chairs_param)
    print(f"Time: {time.time() - ts}")
    if plan is None:
        print("No solution found")
    elif isinstance(plan, PlanningResult):
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.angle_vector(AV_INIT)
        pr2.newcoords(Coordinates([start[0], start[1], 0.0], [start[2], 0.0, 0.0]))
        base_spec = PR2BaseOnlySpec()
        base_spec.set_skrobot_model_state(pr2, plan.base_path_to_pre_remove_chair[0])
        v = PyrenderViewer()
        task.visualize(v)
        v.add(pr2)
        v.show()
        time.sleep(2)
        for q in plan.base_path_to_pre_remove_chair.resample(100):
            base_spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.05)
        input("Press Enter to continue...")

        for q in plan.base_path_to_post_remove_chair.resample(100):
            base_spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.05)

        input("Press Enter to continue...")
        for q in plan.base_path_final:
            base_spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.05)

        input("Press Enter to continue...")
        spec = PR2RarmSpec() if plan.is_rarm else PR2LarmSpec()
        for q in plan.joint_path_final.resample(100):
            spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.05)
        time.sleep(1000)
