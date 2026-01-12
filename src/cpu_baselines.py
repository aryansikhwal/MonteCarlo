import random

def pi_cpu_python(n):
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n

start = time.time()
print("π (CPU Python):", pi_cpu_python(200_000))
print("CPU Python time:", time.time() - start, "seconds")

def pi_cpu_numpy(n):
    x = np.random.rand(n)
    y = np.random.rand(n)
    inside = np.sum(x*x + y*y <= 1)
    return 4 * inside / n

start = time.time()
print("π (CPU NumPy):", pi_cpu_numpy(5_000_000))
print("CPU NumPy time:", time.time() - start, "seconds")
