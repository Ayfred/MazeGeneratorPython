import tracemalloc
import time


class Decorator:
    enable_timer_decorator = False

    @staticmethod
    def timer(func):
        def wrapper(*args, **kwargs):
            if Decorator.enable_timer_decorator:
                start = time.time()
                path, iterations = func(*args, **kwargs)
                end = time.time()
                print(f'Elapsed time: {end - start}')
                return path, iterations
            else:
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def timer_value_iteration(func):
        def wrapper_value_iteration(*args, **kwargs):
            if Decorator.enable_timer_decorator:
                start = time.time()
                path, iterations, values = func(*args, **kwargs)
                end = time.time()
                print(f'Elapsed time: {end - start}')
                return path, iterations, values
            else:
                return func(*args, **kwargs)

        return wrapper_value_iteration

    @staticmethod
    def timer_policy_iteration(func):
        def wrapper_policy_iteration(*args, **kwargs):
            if Decorator.enable_timer_decorator:
                start = time.time()
                path, iterations, values, policy = func(*args, **kwargs)
                end = time.time()
                print(f'Elapsed time: {end - start}')
                return path, iterations, values, policy
            else:
                return func(*args, **kwargs)

        return wrapper_policy_iteration

    @staticmethod
    def memory(func):
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            path, iterations = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return path, iterations, current, peak
            # Return the result and the current memory usage in bytes

        return wrapper

    @staticmethod
    def memory_value_iteration(func):
        def wrapper_value_iteration(*args, **kwargs):
            tracemalloc.start()
            path, iterations, values = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return path, iterations, values, current, peak

        return wrapper_value_iteration

    @staticmethod
    def memory_policy_iteration(func):
        def wrapper_policy_iteration(*args, **kwargs):
            tracemalloc.start()
            path, iterations, values, policy = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return path, iterations, values, policy, current, peak

        return wrapper_policy_iteration

    @staticmethod
    def decorator(func):
        if func.__name__ == 'bfs' or func.__name__ == 'dfs' or func.__name__ == 'a_star':
            def wrapper(*args, **kwargs):
                path, iterations = func(*args, **kwargs)
                return path, iterations

            return wrapper

        elif func.__name__ == 'value_iteration':
            def wrapper_value_iteration(*args, **kwargs):
                path, iterations, values = func(*args, **kwargs)
                return path, iterations, values

            return wrapper_value_iteration
        elif func.__name__ == 'policy_iteration':
            def wrapper_policy_iteration(*args, **kwargs):
                path, iterations, values, policy = func(*args, **kwargs)
                return path, iterations, values, policy

            return wrapper_policy_iteration
        else:
            return None
