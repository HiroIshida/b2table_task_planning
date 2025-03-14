import numpy as np
from plainmp.ik import solve_ik
from plainmp.robot_spec import PR2LarmSpec, PR2RarmSpec
from rpbench.articulated.pr2.thesis_jsk_table import AV_INIT, JskChair
from skrobot.coordinates.math import matrix2quaternion, wxyz2xyzw
from skrobot.model.primitives import Axis
from skrobot.viewers import PyrenderViewer

if __name__ == "__main__":
    np.random.seed(3)
    spec = PR2RarmSpec()
    robot = spec.get_robot_model(with_mesh=True)
    robot.angle_vector(AV_INIT)
    spec.reflect_skrobot_model_to_kin(robot)
    chair = JskChair()

    r_axis = Axis()
    r_axis.translate(
        [-JskChair.DEPTH * 0.5 + 0.04, -JskChair.WIDTH * 0.4, JskChair.BACK_HEIGHT - 0.05]
    )
    r_axis.rotate(0.5 * np.pi, "y")
    r_axis.rotate(0.5 * np.pi, "x")
    chair.assoc(r_axis)
    chair.translate([0.8, 0.0, 0.0])
    robot.assoc(chair)

    cst = spec.create_collision_const()
    sdf = chair.create_sdf()
    cst.set_sdf(sdf)

    pos = r_axis.worldpos()
    quat = wxyz2xyzw(matrix2quaternion(r_axis.worldrot()))
    pose_cst = spec.create_gripper_pose_const(np.hstack([pos, quat]))
    lb, ub = spec.angle_bounds()
    ineq_cst = spec.create_position_bound_const("r_elbow_flex_link", 2, 1.1, 1.2)
    ret = solve_ik(pose_cst, ineq_cst, lb, ub)
    assert ineq_cst.is_valid(ret.q)
    spec.set_skrobot_model_state(robot, ret.q)

    lspec = PR2LarmSpec()
    q_left = ret.q.copy()
    q_left[0] *= -1
    q_left[2] *= -1
    q_left[4] *= -1
    q_left[6] *= -1
    lspec.set_skrobot_model_state(robot, q_left)

    av = robot.angle_vector()
    print(list(av))

    viewer = PyrenderViewer()
    viewer.add(robot)
    viewer.add(r_axis)
    chair.visualize(viewer)
    viewer.show()
    import time

    time.sleep(1000)
