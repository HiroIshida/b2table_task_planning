import numpy as np
import pytest
from rpbench.planer_box_utils import Box2d, PlanerCoords

from b2table_task_planning.cython import compute_box2d_sd


def manual_sdf(points, box_size, box_center):
    return Box2d(extent=box_size, coords=PlanerCoords(pos=box_center, angle=0.0)).sd(points)


@pytest.mark.parametrize(
    "box_size,box_center",
    [
        (np.array([2.0, 2.0]), np.array([0.0, 0.0])),
        (np.array([4.0, 1.0]), np.array([1.0, 2.0])),
        (np.array([3.0, 5.0]), np.array([-2.0, 1.0])),
    ],
)
def test_compute_box2d_sd(box_size, box_center):
    xs = np.linspace(box_center[0] - 5, box_center[0] + 5, 11)
    ys = np.linspace(box_center[1] - 5, box_center[1] + 5, 11)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    expected = manual_sdf(points, box_size, box_center)
    result = compute_box2d_sd(points, box_size, box_center)
    assert np.allclose(result, expected, atol=1e-7)
