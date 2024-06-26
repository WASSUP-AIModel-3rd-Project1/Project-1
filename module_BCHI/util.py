import time
from functools import wraps

def record_time(func):
    @wraps(func) 
    def wrapper(*args,**kwargs):
        start = time.perf_counter()
        rslt = func(*args,**kwargs)
        end = time.perf_counter()
        return rslt, end-start
    return wrapper