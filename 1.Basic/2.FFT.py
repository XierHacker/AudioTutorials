import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def gcd(pair):
    a, b = pair
    low = min(a, b)
    for i in range(low, 0, -1):
        if a % i == 0 and b % i == 0:
            return i

a=list(range(20000))
b=list(range(20000,40000))
pairs=list(zip(a,b))

def compute(pairs):
    # print("pairs:",pairs)
    futures = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for pair in pairs:
            future = pool.submit(gcd, pair)
            futures.append(future)

    result = [future.result() for future in futures]
    end_time = time.time()
    duration = end_time - start_time
    print("spend:", duration)
    print("result:", result)



if __name__=="__main__":
    compute(pairs)

