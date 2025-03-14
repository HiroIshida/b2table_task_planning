cimport numpy as np

import numpy as np

from libc.math cimport fabs, sqrt


def compute_box2d_sd(
    np.ndarray[np.float64_t, ndim=2] points,
    np.ndarray[np.float64_t, ndim=1] box_size,
    np.ndarray[np.float64_t, ndim=1] box_pos):
    cdef Py_ssize_t n = points.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sds = np.empty(n, dtype=np.float64)

    cdef double hx = box_size[0] * 0.5
    cdef double hy = box_size[1] * 0.5

    cdef Py_ssize_t i
    cdef double px, py, dx, dy, outsideDist, insideDist

    for i in range(n):
        px = points[i, 0] - box_pos[0]
        py = points[i, 1] - box_pos[1]
        dx = fabs(px) - hx
        dy = fabs(py) - hy
        outsideDist = sqrt(
            (dx if dx > 0 else 0) * (dx if dx > 0 else 0) +
            (dy if dy > 0 else 0) * (dy if dy > 0 else 0)
        )
        insideDist = min(max(dx, dy), 0)
        sds[i] = outsideDist + insideDist
    return sds
