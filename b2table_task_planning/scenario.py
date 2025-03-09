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


if __name__ == "__main__":
    v = PyrenderViewer()
    task = barely_feasible_task()
    task.visualize(v)
    v.show()
    import time

    time.sleep(1000)
