from typing import Optional

import numpy as np
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from rpbench.articulated.pr2.thesis_jsk_table import JskMessyTableTask, JskTable
from rpbench.planer_box_utils import Box2d, PlanerCoords


class SituationSampler:
    table2d_box: Box2d  # to compute 2d sdf of the table
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

    def sample_pr2_pose(self) -> np.ndarray:
        ...
