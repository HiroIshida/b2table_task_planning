from typing import Optional

import numpy as np
from plainmp.constraint import SphereCollisionCst
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from plainmp.robot_spec import PR2BaseOnlySpec
from rpbench.articulated.pr2.thesis_jsk_table import (
    AV_INIT,
    JskMessyTableTask,
    JskTable,
    fit_radian,
)
from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.planer_box_utils import Box2d, PlanerCoords

from b2table_task_planning.cython import compute_box2d_sd


class SituationSampler:
    table2d_box: Box2d  # to compute 2d sdf of the table
    table_collision_cst: SphereCollisionCst
    target_region: BoxSkeleton  # contains all robot, table, and chairs
    tabletop_obstacle_sdf: Optional[UnionSDF]
    reaching_pose: Optional[np.ndarray]

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
        pr2_spec.get_kin()
        skmodel = pr2_spec.get_robot_model()
        skmodel.angle_vector(AV_INIT)
        pr2_spec.reflect_skrobot_model_to_kin(
            skmodel
        )  # NOTE: robot configuration expect for base is fixed

        cst = pr2_spec.create_collision_const()
        sdf = JskTable().create_sdf()
        cst.set_sdf(sdf)
        self.cst = cst

        # target region
        target_region, table_box2d_wrt_region = JskMessyTableTask._prepare_target_region()
        self.target_region = target_region

        self.tabletop_obstacle_sdf = None
        self.reaching_pose = None

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
        points = self.target_region.sample_points(n_max_trial)

        dists = np.linalg.norm(points[:, :2] - self.reaching_pose[:2], axis=1)
        points_inside = points[dists < 0.8][:, :2]
        yaw_pluss = np.random.uniform(-np.pi / 4, np.pi / 4, size=(len(points_inside), 1))
        yaw_cands = fit_radian(self.reaching_pose[3] + yaw_pluss)
        vector_coords = np.hstack([points_inside, yaw_cands])

        for vector in vector_coords:
            point = vector[:2]
            sd = compute_box2d_sd(
                point.reshape(1, 2), self.table2d_box.extent, self.table2d_box.coords.pos
            )[0]
            if sd > 0.55 or sd < 0.0:
                continue
            if self.cst.is_valid(vector):
                return vector
        return None
