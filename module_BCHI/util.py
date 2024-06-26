import time

def record_time(func):
    start = time.time()
    def wrapper():
        return func
    end = time.time()
    return wrapper, end-start