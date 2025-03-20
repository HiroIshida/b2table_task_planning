import time

import matplotlib.pyplot as plt
import numpy as np

from b2table_task_planning.feasible_checker import (
    FeasibilityChecker,
    FeasibilityCheckerJit,
)

if __name__ == "__main__":
    checker = FeasibilityChecker()
    n_repeat = 100
    bench = {}
    for n_elem in [1, 10, 100, 1000, 2000, 4000]:
        dummy_vectors = np.zeros((n_elem, 7))
        dummy_image1 = np.zeros((112, 112))
        dummy_image2 = np.zeros((112, 112))
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)

        ts = time.time()
        for _ in range(n_repeat):
            checker.infer(dummy_vectors, dummy_image1, dummy_image2)
        elapsed_per_element = (time.time() - ts) / n_elem / n_repeat

        bench[n_elem] = elapsed_per_element

    bench_jit = {}
    for n_elem in [1, 10, 100, 1000, 2000, 4000]:
        checker = FeasibilityCheckerJit(n_elem)
        dummy_vectors = np.zeros((n_elem, 7))
        dummy_image1 = np.zeros((112, 112))
        dummy_image2 = np.zeros((112, 112))
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)

        ts = time.time()
        for _ in range(n_repeat):
            checker.infer(dummy_vectors, dummy_image1, dummy_image2)
        elapsed_per_element = (time.time() - ts) / n_elem / n_repeat

        bench_jit[n_elem] = elapsed_per_element

    print(bench)
    print(bench_jit)

    # log plot
    plt.plot(list(bench.keys()), list(bench.values()), marker="o", label="raw")
    plt.plot(list(bench_jit.keys()), list(bench_jit.values()), marker="o", label="jit")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Batch size")
    plt.ylabel("Time per element [s]")
    plt.grid(which="both")
    plt.savefig("inference_speed_bench.png", dpi=400)
    plt.show()
