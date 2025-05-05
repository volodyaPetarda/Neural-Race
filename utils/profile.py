import time
from functools import wraps


def timeit(N=1):
    def decorator(func):
        call_count = 0
        total_time = 0.0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_count, total_time
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            call_count += 1
            total_time += elapsed

            if call_count % N == 0:
                avg_time = total_time / N
                print(f"{func.__name__} [last {N} calls] "
                      f"Avg: {avg_time:.6f}s | Total: {total_time:.6f}s")
                total_time = 0

            return result

        return wrapper

    return decorator