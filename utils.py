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