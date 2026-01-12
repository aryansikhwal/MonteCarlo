@cuda.jit
def mc_kernel_fast_rng(rng_states, n, counter):
    i = cuda.grid(1)
    if i >= n:
        return

    # use the blockIdx.x RNG state
    bid = cuda.blockIdx.x

    # generate x,y using xoroshiro RNG
    x = xoroshiro128p_uniform_float32(rng_states, bid)
    y = xoroshiro128p_uniform_float32(rng_states, bid)

    if x*x + y*y <= 1.0:
        cuda.atomic.add(counter, 0, 1)


def pi_gpu_fast_rng(n, threads=256):
    blocks = (n + threads - 1) // threads

    # ONLY blocks RNG states, not n states
    rng_states = create_xoroshiro128p_states(blocks, seed=1234)

    counter = cuda.to_device(np.array([0], dtype=np.int32))

    mc_kernel_fast_rng[blocks, threads](rng_states, n, counter)

    inside = counter.copy_to_host()[0]
    return 4 * inside / n
