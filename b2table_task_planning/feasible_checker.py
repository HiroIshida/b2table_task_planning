import copy
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import tqdm
from hifuku.core import SolutionLibrary
from hifuku.domain import Pr2ThesisJskTable2
from hifuku.script_utils import load_library
from plainmp.constraint import SphereCollisionCst
from plainmp.experimental import MultiGoalRRT
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, RefineType
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
from skrobot.model import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from b2table_task_planning.sampler import SituationSampler
from b2table_task_planning.scenario import barely_feasible_task

# fmt: off
AV_CHAIR_GRASP = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.87266463, 0.0, -0.44595566, -0.07283169, -2.2242064, 4.9038143, -1.4365499, -0.93810624, -0.6677667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44595566, -0.07283169, 2.2242064, -4.9038143, -1.4365499, -0.93810624, 0.6677667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# fmt: on

CHAIR_GRASP_BASE_OFFSET = 0.8


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
    remove_chair_idx: Optional[int] = None
    chair_rotation_angle: Optional[float] = None

    def require_repair(self) -> bool:
        return self.base_path_to_pre_remove_chair is not None


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
        region_ub[1] += 1.0
        self.pr2_pose_lb = np.hstack([region_lb[:2], [-np.pi * 1.5]])
        self.pr2_pose_ub = np.hstack([region_ub[:2], [+np.pi * 1.5]])

    def plan(
        self,
        pr2_pose_now: np.ndarray,
        reaching_pose: np.ndarray,
        obstacles_param: np.ndarray,
        chairs_param: np.ndarray,
    ) -> Optional[PlanningResult]:

        # check if pr2_pose_now is inside lb and ub
        if not np.all(pr2_pose_now[:2] > self.pr2_pose_lb[:2]) or not np.all(
            pr2_pose_now[:2] < self.pr2_pose_ub[:2]
        ):
            print("pr2_pose_now is out of the bound")
            return None

        # here we assume that only chair is movable
        self.sampler.register_tabletop_obstacles(obstacles_param)
        self.sampler.register_reaching_pose(reaching_pose)
        self.chair_manager.set_param(chairs_param)
        pr2_pose_cands = self._sample_pr2_pose()
        if pr2_pose_cands is None:
            return None  # no solution found
        recahable_pr2_poses, current_tree = self._select_reachable_poses(
            pr2_pose_now, pr2_pose_cands
        )

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
                current_tree,
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
                current_tree,
            )
            return rplanner.plan(chairs_param)

        print("now the phase of finding feasible solution by actually solving the problem")
        for pose, is_feasible, min_idx in zip(recahable_pr2_poses, is_feasibiles, min_indices):
            if is_feasible:
                task = JskMessyTableTaskWithChair(
                    obstacles_param, chairs_param, pose, reaching_pose
                )
                conf = copy.deepcopy(Pr2ThesisJskTable2.solver_config)
                conf.refine_seq = [RefineType.SHORTCUT, RefineType.BSPLINE]
                solver = Pr2ThesisJskTable2.solver_type.init(Pr2ThesisJskTable2.solver_config)
                solver.setup(task.export_problem())
                init_traj = self.engine.lib.init_solutions[min_idx]
                res = solver.solve(init_traj)
                if res.traj is not None:
                    result = PlanningResult()
                    result.joint_path_final = res.traj
                    result.is_rarm = task.is_using_rarm()
                    result.base_path_final = Trajectory(current_tree.get_solution(pose).T)
                    return result

        assert (
            False
        ), "not supposed to reach here, but there is still very small chance to reach here..."

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
    ) -> Tuple[Optional[np.ndarray], MultiGoalRRT]:
        # check if robot is placable at the sampled pose.
        # Also, return the tree as a bi-product
        sdf = self.chair_manager.create_sdf()
        sdf.merge(self.table.create_sdf())

        pr2_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        skmodel = pr2_spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        pr2_spec.reflect_skrobot_model_to_kin(skmodel)
        cst = pr2_spec.create_collision_const()
        cst.set_sdf(sdf)

        tree = MultiGoalRRT(current_pose, self.pr2_pose_lb, self.pr2_pose_ub, cst, 2000)
        bools = tree.is_reachable_batch(pose_list.T, 0.5)
        if not np.any(bools):
            return None, tree
        return pose_list[bools], tree


def setup_spec_and_coll_cst():
    base_spec = PR2BaseOnlySpec(use_fixed_uuid=True)

    # prepare collision constraint
    skmodel = base_spec.get_robot_model(deepcopy=False)
    skmodel.angle_vector(AV_INIT)
    base_spec.reflect_skrobot_model_to_kin(skmodel)
    collision_cst_base_only = base_spec.create_collision_const()

    # prepare chair-attached collision constraint
    chair_bbox_lb = np.array([-JskChair.DEPTH * 0.5, -JskChair.WIDTH * 0.5, 0.0])
    chair_bbox_ub = np.array([JskChair.DEPTH * 0.5, JskChair.WIDTH * 0.5, JskChair.BACK_HEIGHT])
    x = np.linspace(chair_bbox_lb[0], chair_bbox_ub[0], 10)
    y = np.linspace(chair_bbox_lb[1], chair_bbox_ub[1], 10)
    z = np.linspace(chair_bbox_lb[2], chair_bbox_ub[2], 20)
    xx, yy, zz = np.meshgrid(x, y, z)
    cloud = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    cloud[:, 0] += CHAIR_GRASP_BASE_OFFSET
    aspec = SphereAttachmentSpec("base_footprint", cloud.T, np.ones(len(cloud)) * 0.05, False)
    coll_cst_with_chair = base_spec.create_collision_const(attachements=(aspec,))
    collision_cst_with_chair = coll_cst_with_chair
    return base_spec, collision_cst_base_only, collision_cst_with_chair


_BASE_SPEC, _COLLISION_CST_BASE_ONLY, _COLLISION_CST_WITH_CHAIR = setup_spec_and_coll_cst()


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
    tree_current: MultiGoalRRT
    base_spec: PR2BaseOnlySpec
    collision_cst_base_only: SphereCollisionCst
    collision_cst_with_chair: SphereCollisionCst

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
        tree_current: MultiGoalRRT,
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
        self.tree_current = tree_current

        # do this because creating the spec and collision constraint is expensive
        self.base_spec = _BASE_SPEC
        self.collision_cst_base_only = _COLLISION_CST_BASE_ONLY
        self.collision_cst_with_chair = _COLLISION_CST_WITH_CHAIR

    def plan(self, chairs_param_original: np.ndarray) -> Optional[PlanningResult]:
        n_chair = len(chairs_param_original) // 3
        for i_chair in range(n_chair):
            chair_pose_remove = chairs_param_original[i_chair * 3 : (i_chair + 1) * 3]
            chairs_param_hypo = np.delete(
                chairs_param_original, np.s_[3 * i_chair : 3 * i_chair + 3]
            )
            self.chair_manager.set_param(chairs_param_hypo)
            sdf_hypo = self.chair_manager.create_sdf()
            sdf_hypo.merge(self.table.create_sdf())
            self.collision_cst_base_only.set_sdf(sdf_hypo)

            planning_result = PlanningResult()
            planning_result.remove_chair_idx = i_chair

            # first check if the robot base can reach the goal
            tree_completely_removed = MultiGoalRRT(
                self.init_pr2_pose,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                self.collision_cst_base_only,
                2000,
            )
            bools = tree_completely_removed.is_reachable_batch(self.final_pr2_pose_cands.T, 0.5)
            reachable_poses = self.final_pr2_pose_cands[bools]
            if len(reachable_poses) == 0:
                print(
                    f"giving up the chair {i_chair} because no feasible base pose found even after completely removing the chair"
                )
                continue

            # check if the robot arm can reach the goal pose
            ground_mat = create_map_from_chair_param(chairs_param_hypo)
            gripper_pose_tile = np.tile(self.final_gripper_pose, (reachable_poses.shape[0], 1))
            vectors = np.concatenate([reachable_poses, gripper_pose_tile], axis=1)
            is_feasibiles, min_indices = self.engine.infer(vectors, self.table_mat, ground_mat)
            if not np.any(is_feasibiles):
                print(
                    f"giving up the chair {i_chair} because the arm planning is not feasible even after completely removing the chair"
                )
                continue

            # check if i_chair can be graspable
            ret = self.determine_pregrasp_chair_pr2_base_pose(chair_pose_remove, self.tree_current)
            if ret is None:
                print(f"giving up the chair {i_chair} because grasping base pose is not reachable")
                continue
            pre_remove_pr2_pose, yaw_rot = ret

            planning_result.base_path_to_pre_remove_chair = Trajectory(
                self.tree_current.get_solution(pre_remove_pr2_pose).T
            )
            planning_result.chair_rotation_angle = yaw_rot

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
                print(
                    f"giving up the chair {i_chair} because the arm planning is not 'actually' feasible even after completely removing the chair"
                )
                continue

            # check if feasible placment of the chair is possible
            path = tree_completely_removed.get_solution(feasible_pr2_final_pose).T
            traj = Trajectory(path).resample(100)

            self.collision_cst_with_chair.set_sdf(sdf_hypo)
            pr2_model = self.base_spec.get_robot_model(deepcopy=False)
            pr2_model.angle_vector(AV_CHAIR_GRASP)
            self.base_spec.reflect_skrobot_model_to_kin(pr2_model)
            tree_chair_attach = MultiGoalRRT(
                pre_remove_pr2_pose,
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                self.collision_cst_with_chair,
                2000,
            )
            pr2_model.angle_vector(AV_INIT)
            self.base_spec.reflect_skrobot_model_to_kin(pr2_model)  # reset the kin model
            nodes = tree_chair_attach.get_debug_states()
            dists = np.linalg.norm(nodes[:, :2] - path[-1, :2], axis=1)
            sorted_indices = np.argsort(dists)

            valid_post_remove_pr2_pose = None
            valid_post_remove_chair_pose = None
            for ind in sorted_indices:
                post_remove_pr2_pose = nodes[ind]
                # check if post_remove_pr2_pose is collision free (this is required because we only know that this position is collision free at the grasping pose)
                self.collision_cst_base_only.set_sdf(sdf_hypo)
                if not self.collision_cst_base_only.is_valid(post_remove_pr2_pose):
                    continue

                chair_pose = post_remove_pr2_pose + np.array(
                    [
                        CHAIR_GRASP_BASE_OFFSET * np.cos(post_remove_pr2_pose[2]),
                        CHAIR_GRASP_BASE_OFFSET * np.sin(post_remove_pr2_pose[2]),
                        0.0,
                    ]
                )
                self.chair_manager.set_param(chair_pose)
                sdf_single_chair = self.chair_manager.create_sdf()
                self.collision_cst_base_only.set_sdf(sdf_single_chair)
                is_collision_free = np.all(
                    [self.collision_cst_base_only.is_valid(q) for q in traj.numpy()]
                )
                if not is_collision_free:
                    continue
                valid_post_remove_pr2_pose = post_remove_pr2_pose
                valid_post_remove_chair_pose = chair_pose
                break
            if valid_post_remove_pr2_pose is None:
                print(f"giving up the chair {i_chair} because the chair placement is not feasible")
                continue
            planning_result.base_path_to_post_remove_chair = Trajectory(
                tree_chair_attach.get_solution(valid_post_remove_pr2_pose).T
            )

            # finalizing the plan connecting post_remove_pr2_pose and final_pr2_pose
            chairs_param_post_remove = np.hstack([chairs_param_hypo, valid_post_remove_chair_pose])
            self.chair_manager.set_param(chairs_param_post_remove)
            sdf_post_remove = self.chair_manager.create_sdf()
            sdf_post_remove.merge(self.table.create_sdf())
            self.collision_cst_base_only.set_sdf(sdf_post_remove)

            # assert self.collision_cst_base_only.is_valid(post_remove_pr2_pose)
            assert self.collision_cst_base_only.is_valid(pr2_final_pose)

            ompl_solver_config = OMPLSolverConfig(
                refine_seq=[
                    RefineType.SHORTCUT,
                    RefineType.BSPLINE,
                    RefineType.SHORTCUT,
                    RefineType.BSPLINE,
                ]
            )

            solver = OMPLSolver(ompl_solver_config)
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

            # optionally? smooth result of the base_path_to_pre_remove_chair
            self.collision_cst_base_only.set_sdf(sdf_hypo)
            self.chair_manager.set_param(chairs_param_original)
            sdf_original = self.chair_manager.create_sdf()
            sdf_original.merge(self.table.create_sdf())
            self.collision_cst_base_only.set_sdf(sdf_original)
            solver = OMPLSolver(ompl_solver_config)

            problem = Problem(
                planning_result.base_path_to_pre_remove_chair[0],
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                planning_result.base_path_to_pre_remove_chair[-1],
                self.collision_cst_base_only,
                None,
                np.array([0.1, 0.1, 0.1]),
            )
            ret = solver.solve(problem)
            assert ret.traj is not None, "this should not happen"
            planning_result.base_path_to_pre_remove_chair = ret.traj

            # optionally? smooth result of the base_path_to_post_remove_chair
            self.collision_cst_with_chair.set_sdf(sdf_hypo)
            pr2_model.angle_vector(AV_CHAIR_GRASP)
            self.base_spec.reflect_skrobot_model_to_kin(pr2_model)

            # for q in planning_result.base_path_to_post_remove_chair:
            solver = OMPLSolver(ompl_solver_config)
            problem = Problem(
                planning_result.base_path_to_post_remove_chair[0],
                self.pr2_pose_lb,
                self.pr2_pose_ub,
                planning_result.base_path_to_post_remove_chair[-1],
                self.collision_cst_with_chair,
                None,
                np.array([0.1, 0.1, 0.1]),
            )
            ret = solver.solve(problem)
            assert ret.traj is not None, "this should not happen"
            planning_result.base_path_to_post_remove_chair = ret.traj

            return planning_result
        print("tried to repair the environment but failed")
        return None

    def determine_pregrasp_chair_pr2_base_pose(
        self, chair_pose: np.ndarray, tree: MultiGoalRRT
    ) -> Optional[Tuple[np.ndarray, float]]:
        # check if blocking chair is actually removable by
        # hypothetically chaing the chair yaw angles
        # assuming that PR2 can rotate the chair by 90 degrees
        x, y, yaw = chair_pose
        yaw_rotates = np.linspace(-np.pi * 0.5, np.pi * 0.5, 10)
        yaw_cands = yaw + yaw_rotates
        sins = np.sin(yaw_cands)
        coss = np.cos(yaw_cands)

        xs = x - CHAIR_GRASP_BASE_OFFSET * coss
        ys = y - CHAIR_GRASP_BASE_OFFSET * sins
        pr2_pose_pre_grasp_cands = np.array([xs, ys, yaw_cands]).T
        bools = tree.is_reachable_batch(pr2_pose_pre_grasp_cands.T, 0.5)
        if not np.any(bools):
            return None

        min_yaw = yaw_cands[bools].min()
        x = x - CHAIR_GRASP_BASE_OFFSET * np.cos(min_yaw)
        y = y - CHAIR_GRASP_BASE_OFFSET * np.sin(min_yaw)
        pre_grasp_base_pose = np.array([x, y, min_yaw])
        yaw_rotate = min_yaw - chair_pose[2]
        return pre_grasp_base_pose, yaw_rotate


class SceneVisualizer:
    def __init__(self, init_pr2_pose, final_gripper_pose: np.ndarray, chairs_param: np.ndarray):
        self.table = JskTable()
        self.final_gripper_pose = final_gripper_pose
        self.chair_manager = ChairManager()
        self.chair_manager.set_param(chairs_param)
        self.pr2 = PR2(use_tight_joint_limit=False)
        self.pr2.angle_vector(AV_INIT)
        self.pr2.newcoords(
            Coordinates([init_pr2_pose[0], init_pr2_pose[1], 0.0], [init_pr2_pose[2], 0.0, 0.0])
        )
        self.chair_handles_list = []
        self.v = None

    def visualize(self):
        v = PyrenderViewer()
        self.table.visualize(v)

        axis = Axis()
        gripper_pos, yaw = self.final_gripper_pose[:3], self.final_gripper_pose[3]
        axis.translate(gripper_pos)
        axis.rotate(yaw, "z")
        v.add(axis)

        for i_chair in range(self.chair_manager.n_chair):
            chair = self.chair_manager.chairs[i_chair]
            handles = chair.visualize(v)
            self.chair_handles_list.append(handles)
        v.add(self.pr2)
        v.show()
        self.v = v

    def visualize_base_trajectory(self, traj: Trajectory, fix_chair_idx: Optional[int] = None):
        assert self.v

        if fix_chair_idx is not None:
            self.pr2.angle_vector(AV_CHAIR_GRASP)

        spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        assoc = False
        for q_base in traj.resample(100):
            spec.set_skrobot_model_state(self.pr2, q_base)
            if not assoc and fix_chair_idx is not None:
                vis_prim_handles = self.chair_handles_list[fix_chair_idx]
                for prim in vis_prim_handles:
                    self.pr2.assoc(prim)
                assoc = True
            self.v.redraw()
            time.sleep(0.05)
        if assoc:
            for prim in vis_prim_handles:
                self.pr2.dissoc(prim)
        self.pr2.angle_vector(AV_INIT)

    def visualize_joint_trajectory(self, traj: Trajectory, is_rarm: bool):
        assert self.v
        spec = PR2RarmSpec() if is_rarm else PR2LarmSpec()
        for q_joint in traj.resample(100):
            spec.set_skrobot_model_state(self.pr2, q_joint)
            self.v.redraw()
            time.sleep(0.05)

    def visualize_chair_rotation(self, chair_idx: int, chair_rotation_angle: float):
        assert isinstance(self.v, PyrenderViewer)
        vis_prim_handles = self.chair_handles_list[chair_idx]
        for prim in vis_prim_handles:
            self.v.delete(prim)
        chair = self.chair_manager.chairs[chair_idx]
        chair.rotate(chair_rotation_angle, "z")
        chair_handles = chair.visualize(self.v)
        self.chair_handles_list[chair_idx] = chair_handles
        self.v.redraw()
        time.sleep(0.05)


if __name__ == "__main__":
    task = barely_feasible_task()
    # task = need_fix_task()
    task_planner = TaskPlanner()

    start = np.array([0.784, 2.57, -2.0])
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    plan = task_planner.plan(start, task.reaching_pose, task.obstacles_param, task.chairs_param)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))
    sv = SceneVisualizer(start, task.reaching_pose, task.chairs_param)
    sv.visualize()
    time.sleep(2)
    if plan is None:
        print("No solution found")
        time.sleep(100)
    elif isinstance(plan, PlanningResult):
        print("Solution found")
        input("Press Enter to continue...")
        if plan.require_repair():
            sv.visualize_base_trajectory(plan.base_path_to_pre_remove_chair)
            input("Press Enter to continue...")
            sv.visualize_chair_rotation(plan.remove_chair_idx, plan.chair_rotation_angle)
            input("Press Enter to continue...")
            sv.visualize_base_trajectory(plan.base_path_to_post_remove_chair, plan.remove_chair_idx)
            input("Press Enter to continue...")
        sv.visualize_base_trajectory(plan.base_path_final)
        input("Press Enter to continue...")
        sv.visualize_joint_trajectory(plan.joint_path_final, plan.is_rarm)
        time.sleep(1000)
