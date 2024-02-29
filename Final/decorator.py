import tracemalloc
import functools
import time

"""
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Elapsed time: {end - start}')
        return result
    return wrapper
    """
def memory(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        path, iterations = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f'Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB')
        return path, iterations, current  # Return the result and the current memory usage in bytes
    return wrapper


def decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper


def decorator_factory(*decos):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        for deco in reversed(decos):
            wrapper = deco(wrapper)
        return wrapper
    return decorator
