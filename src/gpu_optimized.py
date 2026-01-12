@cuda.jit
def mc_kernel_optimized(xs, ys, counter):
    i = cuda.grid(1)
    if i < xs.size:
        x = xs[i]
        y = ys[i]
        if x*x + y*y <= 1:
            cuda.atomic.add(counter, 0, 1)

def pi_gpu_optimized(n, threads=256):
    xs = cuda.to_device(np.random.rand(n).astype(np.float32))
    ys = cuda.to_device(np.random.rand(n).astype(np.float32))

    counter = cuda.to_device(np.array([0], dtype=np.int32))

    blocks = (n + threads - 1) // threads

    mc_kernel_optimized[blocks, threads](xs, ys, counter)

    inside = counter.copy_to_host()[0]
    return 4 * inside / n

//streamed version (host device overlap)

def pi_gpu_streamed(total_n, batch_size=4_000_000, threads=256):
    num_batches = total_n // batch_size
    streams = [cuda.stream() for _ in range(3)]

    counters = []
    blocks = (batch_size + threads - 1) // threads

    for b in range(num_batches):
        stream = streams[b % 3]

        # CPU generates batch random numbers WHILE GPU computes other batches
        xs_host = np.random.rand(batch_size).astype(np.float32)
        ys_host = np.random.rand(batch_size).astype(np.float32)

        # allocate device memory for results
        xs_dev = cuda.to_device(xs_host, stream=stream)
        ys_dev = cuda.to_device(ys_host, stream=stream)
        counter = cuda.to_device(np.array([0], dtype=np.int32), stream=stream)

        # launch kernel asynchronously
        mc_kernel_stream[blocks, threads, stream](xs_dev, ys_dev, counter)

        counters.append(counter)

    # synchronization
    for s in streams:
        s.synchronize()

    # combine results
    total_inside = 0
    for c in counters:
        total_inside += c.copy_to_host()[0]

    return 4 * total_inside / total_n
