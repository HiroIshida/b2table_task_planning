import numpy as np
from rpbench.articulated.pr2.thesis_jsk_table import (
    JskMessyTableTaskWithChair,
    JskTable,
)
from skrobot.viewers import PyrenderViewer


def barely_feasible_task():
    pr2_pose = np.zeros(3)  # dummy
    reaching_target = np.array([-0.25, -0.1, JskTable.TABLE_HEIGHT + 0.1, -0.0])
    chairs_param = np.array([-0.7, -0.3, 0.0])
    obstacles_param = np.array([])
    task = JskMessyTableTaskWithChair(obstacles_param, chairs_param, pr2_pose, reaching_target)
    return task


def need_fix_task():
    pr2_pose = np.zeros(3)  # dummy
    reaching_target = np.array([-0.25, +0.25, JskTable.TABLE_HEIGHT + 0.1, -0.7])
    chairs_param = np.array([-0.9, -0.3, 0.0, -0.9, 0.55, 0.2, 0.0, 1.2, -1.8])  # ng
    # chairs_param = np.array([0.0, 1.2, -1.8]) ok
    # chairs_param = np.array([-0.9, 0.55, 0.2, 0.0, 1.2, -1.8]) NG (why??? it's supposed to be ok)
    obstacles_param = np.array([])
    task = JskMessyTableTaskWithChair(obstacles_param, chairs_param, pr2_pose, reaching_target)
    return task


if __name__ == "__main__":
    v = PyrenderViewer()
    task = need_fix_task()
    task.visualize(v)
    v.show()
    import time

    time.sleep(1000)
