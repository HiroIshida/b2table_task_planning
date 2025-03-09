import time

import numpy as np
import torch
import tqdm
from hifuku.core import SolutionLibrary
from hifuku.domain import Pr2ThesisJskTable
from hifuku.script_utils import load_library
from rpbench.articulated.pr2.thesis_jsk_table import JskMessyTableTask, JskTable
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.utils import BoxSkeleton

from jsk_table_task_planning.sampler import SituationSampler


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


class InferenceEngine:
    lib: SolutionLibrary

    def __init__(self):
        self.lib = load_library(Pr2ThesisJskTable, "cuda", postfix="0.2")
        # Not jit compiled intentionally becaues input shape is not fixed
        task = JskMessyTableTask.sample()
        for _ in range(10):  # warm up
            self.lib.infer(task)

    def infer(self, vectors: np.ndarray, table_mat: np.ndarray):
        assert self.lib.ae_model_shared is not None
        vectors_cuda = torch.from_numpy(vectors).float().cuda()
        table_mat_cuda = torch.from_numpy(table_mat).float().cuda().unsqueeze(0).unsqueeze(0)
        encoded = self.lib.ae_model_shared.forward(table_mat_cuda)
        n_batch = vectors_cuda.shape[0]
        encoded_repeated = encoded.repeat(n_batch, 1)

        costs_arr = torch.zeros(len(self.lib.predictors), n_batch).float().cuda()
        for i in range(len(self.lib.predictors)):
            pred = self.lib.predictors[i]
            bias = self.lib.biases[i]
            costs = pred.forward((encoded_repeated, vectors_cuda))[0]
            costs_arr[i] = costs.flatten() + bias
        min_costs, min_indices = torch.min(costs_arr, dim=0)
        return min_costs.cpu().detach().numpy(), min_indices.cpu().detach().numpy()


if __name__ == "__main__":
    sampler = SituationSampler()
    task = JskMessyTableTask.sample()

    engine = InferenceEngine()

    ts = time.time()
    sampler.register_tabletop_obstacles(task.obstacles_param)
    sampler.register_reaching_pose(task.reaching_pose)
    table_mat = create_map_from_obstacle_param(task.obstacles_param)

    pose_list = []
    for _ in tqdm.tqdm(range(1000)):
        pose = sampler.sample_pr2_pose()
        if pose is not None:
            pose_list.append(pose)
    pose_list = np.array(pose_list)

    reaching_pose_tile = np.tile(task.reaching_pose, (pose_list.shape[0], 1))
    vectors = np.concatenate([pose_list, reaching_pose_tile], axis=1)

    min_costs, min_indices = engine.infer(vectors, table_mat)
    print(f"Time: {time.time() - ts:.2f}s")

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.add_patch(
            plt.Rectangle(
                (-0.5 * JskTable.TABLE_DEPTH, -0.5 * JskTable.TABLE_WIDTH),
                JskTable.TABLE_DEPTH,
                JskTable.TABLE_WIDTH,
                fill=False,
            )
        )

        for (x, y, yaw), min_cost in zip(pose_list, min_costs):
            color = "blue" if min_cost < engine.lib.max_admissible_cost else "red"
            dx = np.cos(yaw) * 0.03
            dy = np.sin(yaw) * 0.03
            ax.arrow(x, y, dx, dy, head_width=0.01, length_includes_head=True, color=color)
        plt.axis("equal")
        plt.show()
    else:
        domain = Pr2ThesisJskTable
        solver = domain.solver_type.init(domain.solver_config)

        failure_count = 0
        total_count = 0
        for pose, min_cost, min_idx in zip(pose_list, min_costs, min_indices):
            if min_cost < engine.lib.max_admissible_cost:
                task.pr2_coords = pose
                solver.setup(task.export_problem())
                init_traj = engine.lib.init_solutions[min_idx]
                res = solver.solve(init_traj)
                total_count += 1
                if res.traj is None:
                    failure_count += 1
        print(f"Failure rate: {failure_count / total_count:.2f}")
