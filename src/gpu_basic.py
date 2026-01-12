@cuda.jit
def mc_kernel_basic(xs, ys, results):
    i = cuda.grid(1)
    if i < xs.size:
        x = xs[i]
        y = ys[i]
        results[i] = 1 if x*x + y*y <= 1 else 0

def pi_gpu_basic(n, threads=256):
    xs = cuda.to_device(np.random.rand(n).astype(np.float32))
    ys = cuda.to_device(np.random.rand(n).astype(np.float32))
    results = cuda.device_array(n, dtype=np.int32)

    blocks = (n + threads - 1) // threads

    mc_kernel_basic[blocks, threads](xs, ys, results)
    inside = results.copy_to_host().sum()

    return 4 * inside / n
