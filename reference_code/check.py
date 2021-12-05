import time
from multiprocessing import Pool

import numpy as np


def main():
    xs = np.array(list(range(5000)))

    start = time.time()
    dry = [np.sum([2 * k * xs[k] / (n * (n + 1)) for k in range(1, n)], axis=0) for n in range(1, len(xs))]
    print(f"sync: {time.time() - start}")

    def average_all_before(n, x):
        return np.sum([2 * k * x[k] / (n * (n + 1)) for k in range(1, n)], axis=0)

    start = time.time()

    args = [(n, xs) for n in range(1, len(xs))]
    num_workers = 4
    with Pool(num_workers) as pool:
        results = pool.starmap(lambda x, m: np.sum([2 * k * x[k] / (m * (m + 1)) for k in range(1, m)], axis=0), args)

    print(f"async: {time.time() - start}")
    print(results == dry)


if __name__ == '__main__':
    main()
