import tracemalloc
import time

"""
    Decorator class to add timer and memory usage to the algorithms
    
    The decorator class is used to add the timer and memory usage to the algorithms. The class has 2 static methods:
    - timer: This method is used to add the timer to the algorithms. It takes a function as a parameter and returns a wrapper
    function that calculates the time elapsed to run the algorithm.
    - memory: This method is used to add the memory usage to the algorithms. It takes a function as a parameter and returns a
    wrapper function that calculates the memory usage of the algorithm.

    @Author: Maxime Mu (Ayfred)
    @Date: 2024-02-29
"""


class Decorator:
    # Enable or disable the timer decorator
    enable_timer_decorator = False

    """
        This method is used to add the timer to the algorithms. It takes a function as a parameter and returns a wrapper
        function that calculates the time elapsed to run the algorithm.
        
        @param func: The function to add the timer to
        @return: The wrapper function that calculates the time elapsed to run the algorithm
        """

    @staticmethod
    def timer(func):
        def wrapper(*args, **kwargs):
            if Decorator.enable_timer_decorator:
                start = time.time()
                path = func(*args, **kwargs)
                end = time.time()
                print(f'Elapsed time: {end - start}')
                return path
            else:
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def timer_value_iteration(func):
        def wrapper_value_iteration(*args, **kwargs):
            if Decorator.enable_timer_decorator:
                start = time.time()
                path, values = func(*args, **kwargs)
                end = time.time()
                print(f'Elapsed time: {end - start}')
                return path, values
            else:
                return func(*args, **kwargs)

        return wrapper_value_iteration

    @staticmethod
    def timer_policy_iteration(func):
        def wrapper_policy_iteration(*args, **kwargs):
            if Decorator.enable_timer_decorator:
                start = time.time()
                path, values, policy = func(*args, **kwargs)
                end = time.time()
                print(f'Elapsed time: {end - start}')
                return path, values, policy
            else:
                return func(*args, **kwargs)

        return wrapper_policy_iteration

    """
        This method is used to add the memory usage to the algorithms. It takes a function as a parameter and returns a
        wrapper function that calculates the memory usage of the algorithm.
        
        @param func: The function to add the memory usage to
        @return: The wrapper function that calculates the memory usage of the algorithm
    """

    @staticmethod
    def memory(func):
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            path = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return path, current, peak
            # Return the result and the current memory usage in bytes

        return wrapper

    @staticmethod
    def memory_value_iteration(func):
        def wrapper_value_iteration(*args, **kwargs):
            tracemalloc.start()
            path, values = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return path, values, current, peak

        return wrapper_value_iteration

    @staticmethod
    def memory_policy_iteration(func):
        def wrapper_policy_iteration(*args, **kwargs):
            tracemalloc.start()
            path, values, policy = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return path, values, policy, current, peak

        return wrapper_policy_iteration

    @staticmethod
    def decorator(func):
        if func.__name__ == 'bfs' or func.__name__ == 'dfs' or func.__name__ == 'a_star':
            def wrapper(*args, **kwargs):
                path = func(*args, **kwargs)
                return path

            return wrapper

        elif func.__name__ == 'value_iteration':
            def wrapper_value_iteration(*args, **kwargs):
                path, values = func(*args, **kwargs)
                return path, values

            return wrapper_value_iteration
        elif func.__name__ == 'policy_iteration':
            def wrapper_policy_iteration(*args, **kwargs):
                path, values, policy = func(*args, **kwargs)
                return path, values, policy

            return wrapper_policy_iteration
        else:
            return None
