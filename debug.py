import time

import tqdm
from hifuku.domain import Pr2ThesisJskTable2
from hifuku.script_utils import load_library
from plainmp.robot_spec import PR2LarmSpec, PR2RarmSpec
from rpbench.articulated.pr2.thesis_jsk_table import AV_INIT, JskMessyTableTaskWithChair
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

# algorithm

# np.random.seed(0)


domain = Pr2ThesisJskTable2
conf = domain.solver_config
conf.n_max_call *= 10
solver = domain.solver_type.init(conf)
task = domain.task_type

lib = load_library(domain, "cuda", postfix="0.2")

total = 0
failure = 0
for _ in tqdm.tqdm(range(2000)):
    print("start sampling")
    task = task.sample()
    print("end sampling")
    assert isinstance(task, JskMessyTableTaskWithChair)
    infres = lib.infer(task)
    feasible = infres.cost < lib.max_admissible_cost
    if feasible:
        solver.setup(task.export_problem())
        res = solver.solve(infres.init_solution)

        pr2 = PR2(use_tight_joint_limit=False)
        pr2.angle_vector(AV_INIT)
        co = Coordinates()
        co.translate([task.pr2_coords[0], task.pr2_coords[1], 0])
        co.rotate(task.pr2_coords[2], "z")
        pr2.newcoords(co)

        spec = PR2RarmSpec() if task.is_using_rarm() else PR2LarmSpec()

        v = PyrenderViewer()
        task.visualize(v)
        v.add(pr2)
        v.show()

        for q in res.traj:
            spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.1)

        import time

        time.sleep(1000)
