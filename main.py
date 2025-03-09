from typing import Optional

import numpy as np
from hifuku.domain import Pr2ThesisJskTable
from hifuku.script_utils import load_library
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from rpbench.articulated.pr2.thesis_jsk_table import JskMessyTableTask, JskTable
from rpbench.planer_box_utils import Box2d, PlanerCoords

# v = PyrenderViewer()
# task.visualize(v)
# v.show()
# import time; time.sleep(1000)

obstacles = np.array([])
chairs = np.array([])
pr2_pose = np.zeros(3)
reaching_target = np.array([-0.2, 0.0, 0.8, 0.0])
task = JskMessyTableTask(obstacles, chairs, pr2_pose, reaching_target)
matrix = task.export_task_expression(True).get_matrix()


class SituationStream:
    table2d_box: Box2d  # to compute 2d sdf of the table
    tabletop_obstacle_sdf: Optional[UnionSDF]
    reaching_pose: Optional[np.ndarray]

    def __init__(self):
        table2d_box = Box2d(
            [JskTable.TABLE_DEPTH, JskTable.TABLE_WIDTH], PlanerCoords(np.zeros(2), 0.0)
        )  # global
        self.table2d_box = table2d_box
        self.tabletop_obstacle_sdf = None
        self.reaching_pose = None

    def register_tabletop_obstacles(self, obstacles_param: np.ndarray) -> None:
        n_obstacle = len(obstacles_param) / 6  # x, y, yaw, d, w, h
        tabletop_obstacle_sdfs = []
        for i in range(n_obstacle):
            x, y, yaw, d, w, h = obstacles_param[i * 6 : i * 6 + 6]
            z = JskTable.TABLE_HEIGHT + 0.5 * h
            pose = Pose([x, y, z])
            pose.rotate_z(yaw)
            boxsdf = BoxSDF([d, w, h], pose)
            tabletop_obstacle_sdfs.append(boxsdf)
        self.tabletop_obstacle_sdf = UnionSDF(tabletop_obstacle_sdfs)

    def register_reaching_pose(self, reaching_pose: np.ndarray) -> bool:
        assert len(reaching_pose) == 4
        x, y, z, yaw = reaching_pose
        pts = np.array([[x, y], [x + eps, y], [x, y + eps]])
        sd, sd_pdx, sd_pdy = table2d_box.sd(pts)

        # check if the reaching pose is not too inside the table
        if sd < -0.5:
            return False

        # check if the reaching yaw attack-angle is not too large
        sd_grad = np.array([sd_pdx - sd, sd_pdy - sd])
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        angle = np.arccos(np.dot(sd_grad, direction) / np.linalg.norm(sd_grad))
        if angle > np.pi / 4:
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


domain = Pr2ThesisJskTable
task_type = domain.task_type
lib = load_library(domain, "cuda", postfix="0.2")
lib.jit_compile(batch_predictor=True)

np.random.seed(0)
task = task_type.sample()
# res = task.solve_default()
for _ in range(10):  # warm
    lib.infer(task)

# start
solver = domain.solver_type.init(domain.solver_config)
solver.setup(task.export_problem())


from pyinstrument import Profiler

profiler = Profiler()
profiler.start()
infer_res = lib.infer(task)
solver.setup(task.export_problem())
ret = solver.solve(infer_res.init_solution)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))
print(ret)
