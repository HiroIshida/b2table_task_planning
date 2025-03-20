import time

import matplotlib.pyplot as plt
import numpy as np

from b2table_task_planning.feasible_checker import FeasibilityChecker

if __name__ == "__main__":
    checker = FeasibilityChecker()

    bench = {}

    for n_elem in [10, 100, 1000, 10000]:
        dummy_vectors = np.zeros((n_elem, 7))
        dummy_image1 = np.zeros((112, 112))
        dummy_image2 = np.zeros((112, 112))
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)

        ts = time.time()
        checker.infer(dummy_vectors, dummy_image1, dummy_image2)
        elapsed_per_element = (time.time() - ts) / n_elem

        bench[n_elem] = elapsed_per_element

    # log plot
    plt.plot(list(bench.keys()), list(bench.values()), marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of elements")
    plt.ylabel("Time per element [s]")
    plt.grid(which="both")
    plt.show()
