from typing import Optional

import numpy as np
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from plainmp.robot_spec import PR2BaseOnlySpec
from rpbench.articulated.pr2.thesis_jsk_table import (
    AV_INIT,
    JskMessyTableTask,
    JskTable,
)
from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.planer_box_utils import Box2d, PlanerCoords
from skrobot.model.robot_model import RobotModel

from b2table_task_planning.cpp.sample_pr2_pose import create_sampler, sample_pose


class SituationSampler:
    table2d_box: Box2d  # to compute 2d sdf of the table
    target_region: BoxSkeleton  # contains all robot, table, and chairs
    tabletop_obstacle_sdf: Optional[UnionSDF]
    reaching_pose: Optional[np.ndarray]
    _skmodel: RobotModel

    def __init__(self):
        co2d = PlanerCoords(
            np.array([JskTable.TABLE_DEPTH * 0.5, -JskTable.TABLE_WIDTH * 0.5]), 0.0
        )
        table2d_box = Box2d(
            np.array([JskTable.TABLE_DEPTH * 2, JskTable.TABLE_WIDTH * 2]), co2d
        )  # hypothetical large (doubl size) box
        self.table2d_box = table2d_box

        # setup kin model
        pr2_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        self.reset_kinematics_state()

        cst = pr2_spec.create_collision_const()
        sdf = JskTable().create_sdf()
        cst.set_sdf(sdf)

        # target region
        target_region, table_box2d_wrt_region = JskMessyTableTask._prepare_target_region()
        self.target_region = target_region

        def make_func():
            arr = np.zeros(3)

            def func(x):
                arr[0] = x[0]
                arr[1] = x[1]
                arr[2] = x[2]
                return cst.is_valid(arr)

            return func

        func = make_func()
        self.sampler = create_sampler(
            target_region.extents[:2],
            target_region.worldpos()[:2],
            table2d_box.extent[:2],
            table2d_box.coords.pos,
            func,
            0,
        )

        self.tabletop_obstacle_sdf = None
        self.reaching_pose = None

    def reset_kinematics_state(self):
        pr2_spec = PR2BaseOnlySpec(use_fixed_uuid=True)
        skmodel = pr2_spec.get_robot_model(deepcopy=False)
        skmodel.angle_vector(AV_INIT)
        pr2_spec.reflect_skrobot_model_to_kin(
            skmodel
        )  # NOTE: robot configuration expect for base is fixed
        cst = pr2_spec.create_collision_const()
        sdf = JskTable().create_sdf()
        cst.set_sdf(sdf)

    def register_tabletop_obstacles(self, obstacles_param: np.ndarray) -> bool:
        n_obstacle = int(len(obstacles_param) / 6)  # x, y, yaw, d, w, h
        tabletop_obstacle_sdfs = []
        for i in range(n_obstacle):
            x, y, yaw, d, w, h = obstacles_param[i * 6 : i * 6 + 6]
            # NOTE: dont check if the obstacle is inside the table, because it's obvious
            if d < JskMessyTableTask.OBSTACLE_W_MIN or d > JskMessyTableTask.OBSTACLE_W_MAX:
                return False
            if w < JskMessyTableTask.OBSTACLE_W_MIN or w > JskMessyTableTask.OBSTACLE_W_MAX:
                return False
            if h < JskMessyTableTask.OBSTACLE_H_MIN or h > JskMessyTableTask.OBSTACLE_H_MAX:
                return False

            z = JskTable.TABLE_HEIGHT + 0.5 * h
            pose = Pose([x, y, z])
            pose.rotate_z(yaw)
            boxsdf = BoxSDF([d, w, h], pose)
            tabletop_obstacle_sdfs.append(boxsdf)
        self.tabletop_obstacle_sdf = UnionSDF(tabletop_obstacle_sdfs)
        return True

    def register_reaching_pose(self, reaching_pose: np.ndarray) -> bool:
        assert self.tabletop_obstacle_sdf is not None
        assert len(reaching_pose) == 4
        x, y, z, yaw = reaching_pose
        eps = 1e-3
        pts = np.array([[x, y], [x + eps, y], [x, y + eps]])
        sd, sd_pdx, sd_pdy = self.table2d_box.sd(pts)

        # check if the reaching pose is not too inside the table
        if sd < -0.5:
            return False

        # check if the reaching yaw attack-angle is not too large
        sd_grad = np.array([sd_pdx - sd, sd_pdy - sd])
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        cos_angle = np.dot(sd_grad, direction) / np.linalg.norm(sd_grad)
        if cos_angle > -np.cos(np.pi / 4):  # NOTE: reversed
            return False

        # check if the reacing pose is not too close to the obstacles
        pos = np.array([x, y, z])
        if self.tabletop_obstacle_sdf.evaluate(pos) < 0.03:
            return False

        # check if slided reaching pose is not too close to the obstacles
        pos_slided = pos - np.array([np.cos(yaw), np.sin(yaw), 0.0]) * 0.1
        if self.tabletop_obstacle_sdf.evaluate(pos_slided) < 0.03:
            return False

        self.reaching_pose = reaching_pose
        return True

    def sample_pr2_pose(self, n_max_trial: int = 100) -> Optional[np.ndarray]:
        return sample_pose(self.sampler, self.reaching_pose)
