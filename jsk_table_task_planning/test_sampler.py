import numpy as np
import tqdm
from rpbench.articulated.pr2.thesis_jsk_table import JskMessyTableTask

from jsk_table_task_planning.sampler import SituationSampler


def test_situation_sampler_registration():
    sampler = SituationSampler()
    for _ in range(1000):
        task = JskMessyTableTask.sample()
        assert sampler.register_tabletop_obstacles(task.obstacles_param)
        assert sampler.register_reaching_pose(task.reaching_pose)


def test_situation_sampler_sampling():
    # test identity of sampling distribution (paired with test_situation_sampler_registration)
    sampler = SituationSampler()

    pose_list = []
    for _ in range(1000):
        task = JskMessyTableTask.sample()
        sampler.register_tabletop_obstacles(task.obstacles_param)
        sampler.register_reaching_pose(task.reaching_pose)
        pose = sampler.sample_pr2_pose()
        if pose is not None:
            pose_list.append(pose)
    mean, std = np.mean(pose_list, axis=0), np.std(pose_list, axis=0)

    # sample ground truth
    pose_gt_list = []
    for _ in range(1000):
        task = JskMessyTableTask.sample()
        pose_gt_list.append(task.pr2_coords)
    mean_gt, std_gt = np.mean(pose_gt_list, axis=0), np.std(pose_gt_list, axis=0)

    print(f"mean: {mean}, std: {std}")
    print(f"mean_gt: {mean_gt}, std_gt: {std_gt}")

    assert np.allclose(mean, mean_gt, atol=1e-1), f"mean: {mean}, mean_gt: {mean_gt}"
    assert np.allclose(std, std_gt, atol=1e-1), f"std: {std}, std_gt: {std_gt}"


if __name__ == "__main__":
    sampler = SituationSampler()
    pose_list = []
    for _ in tqdm.tqdm(range(100)):
        task = JskMessyTableTask.sample()
        sampler.register_tabletop_obstacles(task.obstacles_param)
        sampler.register_reaching_pose(task.reaching_pose)
        pose = sampler.sample_pr2_pose()
        if pose is not None:
            pose_list.append(pose)
    pose_arr = np.array(pose_list)
    mean, std = np.mean(pose_arr, axis=0), np.std(pose_arr, axis=0)
    print(mean)
    print(std)
