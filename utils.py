import random
import time
from functools import wraps


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class RandomIter:
    def __init__(self, top_iter, rand_bound, seed=0):
        self.rand_bound = rand_bound
        self.top_iter = top_iter
        self.index = 0
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed)
        return self

    def __next__(self):
        if self.index == self.top_iter:
            raise StopIteration
        self.index += 1
        return random.randint(0, self.rand_bound)

def cluster_patterns(patterns: list[list[int]], similarity_threshold=0.8):
    clusters = [[pattern] for pattern in patterns]

    for i in range(len(patterns)):
        # cause the patterns will be removed into clusters
        if i == len(clusters):
            break

        cluster_expanded = 0
        while cluster_expanded > -1:
            cluster_expanded = -1

            for pattern_id, pattern in enumerate(clusters[i][cluster_expanded+1:]):
                for pattern_target in clusters[i+1:]:
                    if similarity_of_lists(pattern, pattern_target[0]) >= similarity_threshold:
                        expansion = pattern_target[0].copy()
                        clusters.remove(pattern_target)
                        clusters[i].append(expansion)
                        cluster_expanded = pattern_id

    return clusters




def similarity_of_lists(list_1, list_2):
    if len(list_1) != len(list_2):
        return False
    if len(list_1) == 0 or len(list_2) == 0:
        raise ValueError("Similarity list lengths == 0")
    similar = 0
    for el1, el2 in zip(list_1, list_2):
        if el1 == el2:
            similar += 1
    return similar / len(list_1)


def insert_means_floored(lst):
    # Create a new list to store the result
    result = []

    # Iterate over the original list
    for i in range(len(lst) - 1):
        # Append the current element
        result.append(lst[i])
        # Calculate the mean of the current element and the next element
        mean_value = (lst[i] + lst[i + 1]) // 2
        # Append the mean value
        result.append(mean_value)

    # Append the last element of the original list
    result.append(lst[-1])

    return result


if __name__ == '__main__':
    test_patterns = [
    [3, 1, 4, 6, 0, 7, 9, 5],
    [3, 1, 4, 6, 0, 7, 9, 5],
    [8, 0, 9, 7, 4, 3, 5, 6],
    [3, 1, 4, 6, 0, 7, 9, 1],
    [3, 1, 4, 6, 0, 7, 9, 0],
    [8, 1, 2, 0, 6, 5, 9, 7],
    [7, 2, 6, 1, 4, 3, 5, 0],
    [7, 2, 6, 1, 4, 3, 5, 0],

    ]
    cluster_patterns(test_patterns)
