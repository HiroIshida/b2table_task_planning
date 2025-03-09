from typing import Optional

import numpy as np
from plainmp.psdf import BoxSDF, Pose, UnionSDF
from rpbench.articulated.pr2.thesis_jsk_table import JskMessyTableTask, JskTable
from rpbench.planer_box_utils import Box2d, PlanerCoords


def test_situation_sampler():
    sampler = SituationSampler()
    for _ in range(1000):
        task = JskMessyTableTask.sample()
        assert sampler.register_tabletop_obstacles(task.obstacles_param)
        assert sampler.register_reaching_pose(task.reaching_pose)
