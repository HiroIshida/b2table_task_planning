import copy
import hashlib
import pickle
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from hifuku.core import SolutionLibrary
from hifuku.domain import Pr2ThesisJskTable2
from hifuku.script_utils import load_library
from plainmp.constraint import SphereCollisionCst
from plainmp.experimental import MultiGoalRRT
from plainmp.ompl_solver import (
    OMPLSolver,
    OMPLSolverConfig,
    RefineType,
    set_random_seed,
)
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
from skrobot.model import Axis, RobotModel
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from b2table_task_planning.sampler import SituationSampler
from b2table_task_planning.scenario import need_fix_task

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
        print(f"num vector {len(vectors)}")
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


class FeasibilityCheckerJit:
    lib: SolutionLibrary

    def __init__(self, n_batch: int):
        self.lib = load_library(Pr2ThesisJskTable2, "cuda", postfix="0.2")
        self.lib.jit_compile(True, n_batch)
        self.dummy_vector = torch.zeros(n_batch, 7).float().cuda()
        self.biases = torch.Tensor(self.lib.biases).float().cuda()

        # warm up
        vectors = np.random.randn(1, 7).astype(np.float32)
        table_mat = np.random.randn(112, 112).astype(np.float32)
        ground_mat = np.random.randn(112, 112).astype(np.float32)
        for _ in range(10):
            self.infer(vectors, table_mat, ground_mat)

    def infer(self, vectors: np.ndarray, table_mat: np.ndarray, ground_mat: np.ndarray):
        print(f"num vector {len(vectors)}")
        vectors = torch.from_numpy(vectors).float().cuda()
        table_mat = torch.from_numpy(table_mat).float().cuda()  # 112 x 112
        ground_mat = torch.from_numpy(ground_mat).float().cuda()  # 112 x 112

        # first pass the CNN model
        mat = torch.stack([table_mat, ground_mat], dim=0).unsqueeze(0)
        encoded = self.lib.ae_model_shared.forward(mat)

        # then FCN
        n_vector = vectors.shape[0]
        self.dummy_vector[:n_vector] = vectors
        encoded.repeat(self.dummy_vector.shape[0], 1)

        costs = self.lib.batch_predictor(encoded, self.dummy_vector)
        cost_calibrated = costs[:n_vector] + self.biases
        # cost_calibrated has (n_batch, n_predictor)
        # now for each inference among the btach we take fthe minimum cost of predictor output
        min_costs, min_indices = torch.min(cost_calibrated, dim=1)
        return (
            min_costs.cpu().detach().numpy() < self.lib.max_admissible_cost,
            min_indices.cpu().detach().numpy(),
        )


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

    def hash_value(self) -> str:
        def _hash(obj) -> str:
            if obj is None:
                return "None"
            return hashlib.md5(pickle.dumps(obj)).hexdigest()[:8]

        parts = [
            _hash(self.joint_path_final),
            _hash(self.is_rarm),
            _hash(self.base_path_final),
            _hash(self.base_path_to_post_remove_chair),
            _hash(self.base_path_to_pre_remove_chair),
            _hash(self.remove_chair_idx),
            _hash(self.chair_rotation_angle),
        ]
        return "-".join(parts)


class CommonResource:
    engine: FeasibilityChecker
    sampler: SituationSampler
    chair_manager: ChairManager
    table: JskTable
    pr2_pose_lb: np.ndarray
    pr2_pose_ub: np.ndarray
    base_spec: PR2BaseOnlySpec
    pr2_model: RobotModel
    collision_cst_base_only: SphereCollisionCst
    collision_cst_with_chair: SphereCollisionCst

    def __init__(self):
        self.sampler = SituationSampler()
        self.chair_manager = ChairManager()
        self.engine = FeasibilityCheckerJit(1000)
        self.table = JskTable()

        # define the bound of the pr2 pose
        target_region, _ = JskMessyTableTaskWithChair._prepare_target_region()
        region_lb = target_region.worldpos() - 0.5 * target_region.extents
        region_ub = target_region.worldpos() + 0.5 * target_region.extents
        region_ub[1] += 1.0
        self.pr2_pose_lb = np.hstack([region_lb[:2], [-np.pi * 1.5]])
        self.pr2_pose_ub = np.hstack([region_ub[:2], [+np.pi * 1.5]])

        # prepare spec
        base_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        self.base_spec = base_spec
        self.pr2_model = base_spec.get_robot_model(deepcopy=False)

        # prepare collision constraint
        skmodel = base_spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        base_spec.reflect_skrobot_model_to_kin(skmodel)
        collision_cst_base_only = base_spec.create_collision_const()
        self.collision_cst_base_only = collision_cst_base_only

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
        self.collision_cst_with_chair = collision_cst_with_chair

    def create_sdf_table_and_chairs(self, chairs_param: np.ndarray) -> UnionSDF:
        self.chair_manager.set_param(chairs_param)
        sdf = self.chair_manager.create_sdf()
        sdf.merge(self.table.create_sdf())
        return sdf

    def solve_base_motion_planning(
        self, start: np.ndarray, goal: np.ndarray, sdf: UnionSDF, grasping_chair: bool
    ) -> Trajectory:
        # here, we assume that the problem is feasible
        if grasping_chair:
            angle_vector = AV_CHAIR_GRASP
            cst = self.collision_cst_with_chair
        else:
            angle_vector = AV_INIT
            cst = self.collision_cst_base_only

        cst.set_sdf(sdf)
        self.pr2_model.angle_vector(angle_vector)
        self.base_spec.reflect_skrobot_model_to_kin(self.pr2_model)

        ompl_solver_config = OMPLSolverConfig(
            refine_seq=[
                RefineType.SHORTCUT,
                RefineType.SHORTCUT,
                RefineType.BSPLINE,
            ]
        )

        solver = OMPLSolver(ompl_solver_config)
        problem = Problem(
            start,
            self.pr2_pose_lb,
            self.pr2_pose_ub,
            goal,
            cst,
            None,
            np.array([0.1, 0.1, 0.1]),
        )
        ret = solver.solve(problem)
        assert ret.traj is not None, "this should not happen"
        return ret.traj

    def build_base_motion_tree(
        self, current_pose: np.ndarray, sdf: UnionSDF, grasping_chair: bool
    ) -> MultiGoalRRT:
        if grasping_chair:
            angle_vector = AV_CHAIR_GRASP
            cst = self.collision_cst_with_chair
        else:
            angle_vector = AV_INIT
            cst = self.collision_cst_base_only
        cst.set_sdf(sdf)

        self.pr2_model.angle_vector(angle_vector)
        self.base_spec.reflect_skrobot_model_to_kin(self.pr2_model)
        tree = MultiGoalRRT(current_pose, self.pr2_pose_lb, self.pr2_pose_ub, cst, 2000)
        return tree


@dataclass
class PlanningProblem:
    pr2_pose_now: np.ndarray
    reaching_pose: np.ndarray
    obstacles_param: np.ndarray
    chairs_param: np.ndarray


class TaskPlanner:
    common: CommonResource

    def __init__(self):
        self.common = CommonResource()

    def plan(self, problem: PlanningProblem) -> Optional[PlanningResult]:
        # check if pr2_pose_now is inside lb and ub
        if not np.all(problem.pr2_pose_now[:2] > self.common.pr2_pose_lb[:2]) or not np.all(
            problem.pr2_pose_now[:2] < self.common.pr2_pose_ub[:2]
        ):
            print("pr2_pose_now is out of the bound")
            return None

        # here we assume that only chair is movable
        self.common.sampler.register_tabletop_obstacles(problem.obstacles_param)
        self.common.sampler.register_reaching_pose(problem.reaching_pose)
        self.common.chair_manager.set_param(problem.chairs_param)
        pr2_pose_cands = self._sample_pr2_pose()
        if pr2_pose_cands is None:
            return None  # no solution found
        sdf_now = self.common.create_sdf_table_and_chairs(problem.chairs_param)
        tree_now = self.common.build_base_motion_tree(
            problem.pr2_pose_now, sdf_now, grasping_chair=False
        )
        bools = tree_now.is_reachable_batch(pr2_pose_cands.T, 0.5)

        if not np.any(bools):
            rplanner = RepairPlanner(
                problem,
                self.common,
                pr2_pose_cands,
                create_map_from_obstacle_param(problem.obstacles_param),
                tree_now,
            )
            return rplanner.plan(problem.chairs_param)

        # finally
        reachable_pr2_poses = pr2_pose_cands[bools]
        table_mat = create_map_from_obstacle_param(problem.obstacles_param)
        ground_mat = create_map_from_chair_param(problem.chairs_param)

        reaching_pose_tile = np.tile(problem.reaching_pose, (reachable_pr2_poses.shape[0], 1))
        vectors = np.concatenate([reachable_pr2_poses, reaching_pose_tile], axis=1)
        is_feasibiles, min_indices = self.common.engine.infer(vectors, table_mat, ground_mat)

        # check if any feasible solution exists
        if not np.any(is_feasibiles):
            rplanner = RepairPlanner(
                problem,
                self.common,
                pr2_pose_cands,
                table_mat,
                tree_now,
            )
            return rplanner.plan(problem.chairs_param)

        print("now the phase of finding feasible solution by actually solving the problem")
        for pose, is_feasible, min_idx in zip(reachable_pr2_poses, is_feasibiles, min_indices):
            if is_feasible:
                task = JskMessyTableTaskWithChair(
                    problem.obstacles_param, problem.chairs_param, pose, problem.reaching_pose
                )
                conf = copy.deepcopy(Pr2ThesisJskTable2.solver_config)
                conf.refine_seq = [RefineType.SHORTCUT, RefineType.BSPLINE]
                solver = Pr2ThesisJskTable2.solver_type.init(Pr2ThesisJskTable2.solver_config)
                solver.setup(task.export_problem())
                init_traj = self.common.engine.lib.init_solutions[min_idx]
                res = solver.solve(init_traj)
                if res.traj is not None:
                    result = PlanningResult()
                    result.joint_path_final = res.traj
                    result.is_rarm = task.is_using_rarm()
                    result.base_path_final = Trajectory(tree_now.get_solution(pose).T)
                    return result

        assert (
            False
        ), "not supposed to reach here, but there is still very small chance to reach here..."

    def _sample_pr2_pose(self) -> Optional[np.ndarray]:
        pose_list = []
        n_sample_pose = 1000  # temp
        for _ in range(n_sample_pose):
            pose = self.common.sampler.sample_pr2_pose()
            if pose is not None:
                pose_list.append(pose)
        if len(pose_list) == 0:
            return None  # no solution found
        pose_list = np.array(pose_list)
        return pose_list


class RepairPlanner:
    common: CommonResource
    problem: PlanningProblem
    final_pr2_pose_cands: np.ndarray
    table_mat: np.ndarray  # heightmap of the table (can be genrated from obstacle_param but we cache it)
    tree_current: MultiGoalRRT

    def __init__(
        self,
        problem: PlanningProblem,
        common: CommonResource,
        final_pr2_pose_cands: np.ndarray,
        table_mat: np.ndarray,
        tree_current: MultiGoalRRT,
    ):
        self.problem = problem
        self.common = common
        self.final_pr2_pose_cands = final_pr2_pose_cands
        self.table_mat = table_mat
        self.tree_current = tree_current
        self.sdf_original = self.common.create_sdf_table_and_chairs(problem.chairs_param)

    def plan(self, chairs_param_original: np.ndarray) -> Optional[PlanningResult]:

        n_chair = len(chairs_param_original) // 3
        for i_chair in range(n_chair):
            result = self.plan_single(chairs_param_original, i_chair)
            if result is not None:
                return result
        return None

    def plan_single(
        self, chairs_param_now: np.ndarray, remove_chair_idx: int
    ) -> Optional[PlanningResult]:
        print(f"trying hypothetical repair for chair {remove_chair_idx}")
        chair_pose_remove = chairs_param_now[remove_chair_idx * 3 : (remove_chair_idx + 1) * 3]
        chairs_param_hypo = np.delete(
            chairs_param_now, np.s_[3 * remove_chair_idx : 3 * remove_chair_idx + 3]
        )
        planning_result = PlanningResult()
        planning_result.remove_chair_idx = remove_chair_idx

        # check if i_chair can be graspable
        success = self.plan_base_motion_to_chair_grasp_pose(
            chair_pose_remove, self.tree_current, planning_result
        )
        if not success:
            return None

        # build tree for the hypothetical environment
        sdf_hypo = self.common.create_sdf_table_and_chairs(chairs_param_hypo)
        tree_hypo = self.common.build_base_motion_tree(
            self.problem.pr2_pose_now, sdf_hypo, grasping_chair=False
        )

        # check if plan is feasible after removing the chair
        success = self.plan_base_and_arm_hypo_removed(
            tree_hypo, chairs_param_hypo, chair_pose_remove, planning_result
        )
        if not success:
            return None
        feasible_pr2_final_pose = planning_result.base_path_final[-1]

        # check if feasible placment of the chair is possible
        success = self.plan_base_motion_move_chair(sdf_hypo, planning_result)
        if not success:
            return None
        valid_post_remove_chair_pose = planning_result.tmp  # FIXME: this is temporary

        # finalizing the plan connecting post_remove_pr2_pose and final_pr2_pose
        sdf_post_remove = self.common.create_sdf_table_and_chairs(
            np.hstack([chairs_param_hypo, valid_post_remove_chair_pose])
        )
        planning_result.base_path_final = self.common.solve_base_motion_planning(
            planning_result.base_path_to_post_remove_chair[-1],
            feasible_pr2_final_pose,
            sdf_post_remove,
            grasping_chair=False,
        )

        # optionally? sommoth out the trajectories which are already feasible
        planning_result.base_path_to_pre_remove_chair = self.common.solve_base_motion_planning(
            planning_result.base_path_to_pre_remove_chair[0],
            planning_result.base_path_to_pre_remove_chair[-1],
            self.sdf_original,
            grasping_chair=False,
        )

        planning_result.base_path_to_post_remove_chair = self.common.solve_base_motion_planning(
            planning_result.base_path_to_post_remove_chair[0],
            planning_result.base_path_to_post_remove_chair[-1],
            sdf_hypo,
            grasping_chair=True,
        )
        return planning_result

    def plan_base_motion_to_chair_grasp_pose(
        self, chair_pose: np.ndarray, tree: MultiGoalRRT, result: PlanningResult
    ) -> bool:

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
            print("no reachable pre-grasp base pose found")
            return False

        min_yaw = yaw_cands[bools].min()
        x = x - CHAIR_GRASP_BASE_OFFSET * np.cos(min_yaw)
        y = y - CHAIR_GRASP_BASE_OFFSET * np.sin(min_yaw)
        pre_grasp_base_pose = np.array([x, y, min_yaw])
        yaw_rotate = min_yaw - chair_pose[2]
        result.chair_rotation_angle = yaw_rotate

        result.base_path_to_pre_remove_chair = Trajectory(tree.get_solution(pre_grasp_base_pose).T)
        return True

    def plan_base_and_arm_hypo_removed(
        self,
        tree_hypo: MultiGoalRRT,
        chairs_param_hypo: np.ndarray,
        chair_pose: np.ndarray,
        result: PlanningResult,
    ) -> bool:
        bools = tree_hypo.is_reachable_batch(self.final_pr2_pose_cands.T, 0.5)
        reachable_poses = self.final_pr2_pose_cands[bools]
        if len(reachable_poses) == 0:
            print("no feasible base pose found even after completely removing the chair")
            return False

        ground_mat = create_map_from_chair_param(chairs_param_hypo)
        gripper_pose_tile = np.tile(self.problem.reaching_pose, (reachable_poses.shape[0], 1))
        vectors = np.concatenate([reachable_poses, gripper_pose_tile], axis=1)
        is_feasibiles, min_indices = self.common.engine.infer(vectors, self.table_mat, ground_mat)
        if not np.any(is_feasibiles):
            print("the arm planning is not feasible even after completely removing the chair")
            return False

        feasible_pr2_final_pose = None
        for pr2_final_pose, is_feasible, min_idx in zip(
            reachable_poses, is_feasibiles, min_indices
        ):
            if not is_feasible:
                continue
            reaching_task = JskMessyTableTaskWithChair(
                self.problem.obstacles_param,
                chairs_param_hypo,
                pr2_final_pose,
                self.problem.reaching_pose,
            )
            solver = Pr2ThesisJskTable2.solver_type.init(Pr2ThesisJskTable2.solver_config)
            solver.setup(reaching_task.export_problem())
            init_traj = self.common.engine.lib.init_solutions[min_idx]
            res = solver.solve(init_traj)
            if res.traj is None:
                continue
            feasible_pr2_final_pose = pr2_final_pose
            result.joint_path_final = res.traj
            result.is_rarm = reaching_task.is_using_rarm()
            break
        if feasible_pr2_final_pose is None:
            print("no feasible solution found")
            return False

        # NOTE: this path is temporary and will be replaced by the actual path
        result.base_path_final = Trajectory(tree_hypo.get_solution(feasible_pr2_final_pose).T)
        return True

    def plan_base_motion_move_chair(self, sdf_hypo: UnionSDF, result: PlanningResult) -> bool:
        assert result.base_path_final is not None
        base_final_path_tentative = result.base_path_final.resample(100).numpy()

        assert result.base_path_to_pre_remove_chair is not None
        tree_chair_attach = self.common.build_base_motion_tree(
            result.base_path_to_pre_remove_chair[-1], sdf_hypo, grasping_chair=True
        )

        self.common.pr2_model.angle_vector(AV_INIT)
        self.common.base_spec.reflect_skrobot_model_to_kin(
            self.common.pr2_model
        )  # reset the kin model
        nodes = tree_chair_attach.get_debug_states()
        dists = np.linalg.norm(nodes[:, :2] - base_final_path_tentative[-1, :2], axis=1)
        sorted_indices = np.argsort(dists)

        valid_post_remove_pr2_pose = None
        valid_post_remove_chair_pose = None
        for ind in sorted_indices:
            post_remove_pr2_pose = nodes[ind]
            # check if post_remove_pr2_pose is collision free (this is required because we only know that this position is collision free at the grasping pose)
            self.common.collision_cst_base_only.set_sdf(sdf_hypo)
            if not self.common.collision_cst_base_only.is_valid(post_remove_pr2_pose):
                continue

            chair_pose = post_remove_pr2_pose + np.array(
                [
                    CHAIR_GRASP_BASE_OFFSET * np.cos(post_remove_pr2_pose[2]),
                    CHAIR_GRASP_BASE_OFFSET * np.sin(post_remove_pr2_pose[2]),
                    0.0,
                ]
            )
            self.common.chair_manager.set_param(chair_pose)
            sdf_single_chair = self.common.chair_manager.create_sdf()
            self.common.collision_cst_base_only.set_sdf(sdf_single_chair)

            is_collision_free = True
            for q in base_final_path_tentative:
                if not self.common.collision_cst_base_only.is_valid(q):
                    is_collision_free = False
                    break
            if not is_collision_free:
                continue

            valid_post_remove_pr2_pose = post_remove_pr2_pose
            valid_post_remove_chair_pose = chair_pose
            break
        if valid_post_remove_pr2_pose is None:
            print("no feasible solution found")
            return False
        result.base_path_to_post_remove_chair = Trajectory(
            tree_chair_attach.get_solution(valid_post_remove_pr2_pose).T
        )
        result.tmp = valid_post_remove_chair_pose
        return True


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
    np.random.seed(1)
    set_random_seed(0)
    # task = need_dual_fix_task()
    task = need_fix_task()
    task_planner = TaskPlanner()
    problem = PlanningProblem(
        np.array([0.784, 2.57, -2.0]),
        task.reaching_pose,
        task.obstacles_param,
        task.chairs_param,
    )
    ts = time.time()
    plan = task_planner.plan(problem)
    print(f"elapsed time: {time.time() - ts:.3f} [sec]")
    print(f"hash value: {plan.hash_value()}")

    sv = SceneVisualizer(problem.pr2_pose_now, task.reaching_pose, task.chairs_param)
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
