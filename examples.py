import time
import kiddo_py
import numpy as np


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} @ {end - start:.1f}s")
        return result

    return wrapper


if __name__ == "__main__":
    dim = 2  # 2 or 3

    rng = np.random.default_rng(seed=0)
    points = rng.random((10000, dim)).astype(np.float32)
    query_points = rng.random((10000, dim)).astype(np.float32)

    @timer
    def create_tree():
        tree = kiddo_py.PyKdTree(dimensions=dim, points=points)
        # points are indexed as 0, 1, 2, ... based on row position
        # tree is immutable, and all points must be provided at construction time
        # this gives better performance and to avoid issues with duplicate coordinate values

        return tree

    tree = create_tree()

    @timer
    def within_unsorted():
        results = tree.within_unsorted(distance=0.5, query_points=query_points, parallel=True)
        # query_points are indexed as 0, 1, 2, ... based on row position
        # parallel is optional and defaults to False. recommended to turn on for large datasets

        print(results.shape)  # (48611794, 3), each row is [query_index, point_index, distance]

    within_unsorted()

    @timer
    def query_pairs():
        results = tree.query_pairs(distance=0.5, parallel=True)
        # parallel is optional and defaults to False. recommended to turn on for large datasets

        print(results.shape)  # (24328694, 3), each row is [point_index_i, point_index_j, distance]

    query_pairs()
