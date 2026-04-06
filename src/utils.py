def measure_runtime(func):
  import time
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Function {func.__name__} took: {round(end - start, 4)} seconds")
    return result

  return wrapper
