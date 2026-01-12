# GPU-Accelerated Monte Carlo Simulation for π Estimation (CUDA + Numba)

This project implements **Monte Carlo simulation** to estimate the value of **π**, and progressively optimizes it from a CPU baseline to a high-performance GPU implementation using **CUDA via Numba**.

The main highlight is **performance engineering**: starting from a correct baseline, identifying bottlenecks, and iterating through multiple GPU optimizations until achieving significant acceleration using **true GPU-side random number generation (RNG)**.

---

##  Project Goals

- Estimate π using Monte Carlo sampling:
  \[
  \pi \approx 4 \cdot \frac{\#\{(x,y): x^2 + y^2 \le 1\}}{N}
  \]
- Compare execution performance across:
  - CPU Python baseline
  - CPU NumPy optimized baseline
  - GPU naive implementation (host-generated randoms)
  - GPU optimized kernel + reduction
  - GPU-side RNG implementation (major speed-up)

---

##  Architecture / Optimization Journey


### 1) CPU Baseline (Python loops)
- Function: `pi_cpu_python(n)`
- Uses Python `random.random()` in a loop
- Establishes a correctness baseline, but slow due to Python overhead

---

### 2) CPU Optimized Baseline (Vectorized NumPy)
- Function: `pi_cpu_numpy(n)`
- Uses vectorized NumPy random generation and math operations
- Strong baseline used to compare GPU performance fairly

---

### 3) GPU Basic Version (Naive CUDA Kernel)
- Kernel: `mc_kernel_basic`
- Function: `pi_gpu_basic(n)`

**Approach**
- Generate `x` and `y` using `np.random.rand()` on CPU
- Copy arrays to GPU (`cuda.to_device`)
- GPU kernel checks inside-circle condition per sample

**Issue**
- Major bottleneck: **host-side random generation + PCIe transfer**
- GPU compute is fast, but overall time suffers due to CPU↔GPU overhead

---

### 4) GPU Optimized Kernel (Reduction + Better Memory Use)
- Kernel: `mc_kernel_optimized`
- Function: `pi_gpu_optimized(n)`

**Improvements**
- Reduced global memory usage by aggregating counts efficiently
- Optimized kernel logic for better throughput

**Still bottlenecked**
Even with a faster kernel, performance is limited because:
- randomness is still generated on CPU
- data transfer dominates runtime

This is the key engineering insight:  
-- **GPU acceleration is useless if the CPU remains the bottleneck.**

---

### 5) True GPU-Side RNG (Final High-Performance Version)
- Kernel: `mc_kernel_fast_rng`
- Function: `pi_gpu_fast_rng(n)`
- Uses:
  - `create_xoroshiro128p_states`
  - `xoroshiro128p_uniform_float32`

**Why this is the turning point**
- Random numbers are generated directly **inside GPU threads**
- No large host arrays
- No CPU RNG bottleneck
- No massive PCIe transfers

This finally unlocks the true advantage of GPU parallelism.

---

##  Key CUDA / Parallel Concepts Demonstrated

- **CUDA thread indexing** (`cuda.grid(1)`)
- Parallel Monte Carlo sampling
- GPU memory allocations (`device_array`, device counters)
- Kernel optimization (bottleneck-driven)
- Reduction / aggregation strategy for inside-circle counts
- **True GPU-side RNG using xoroshiro128p**

---

##  Repository Structure (Recommended)

```bash
gpu-monte-carlo/
├── notebooks/
│   └── GPUacc_montecarlo.ipynb
├── src/                     
│   ├── cpu_baselines.py
│   ├── gpu_basic.py
│   ├── gpu_optimized.py
│   └── gpu_fast_rng.py
├── results/
│   ├── performance_plot.png
│   └── metrics.txt
└── README.md
```

---

##  Requirements

- Python 3.x
- Numba
- CUDA-capable GPU (recommended)

Install dependencies:

```bash
pip install numba numpy matplotlib
```

 **This project uses `numba.cuda`, which requires a CUDA-enabled setup for GPU execution.

---

##  How to Run

Open and run the notebook:

```bash
notebooks/GPUacc_montecarlo.ipynb
```

The notebook runs experiments in the same order as the optimization journey:
1. CPU Python
2. CPU NumPy
3. GPU basic
4. GPU optimized
5. GPU with true GPU-side RNG

It also generates performance plots comparing runtimes for different `N`.

---

##  Output / Benchmarking

The notebook benchmarks runtime for multiple input sizes, such as:
- 2M, 5M, 10M samples
- 5M, 10M, 20M samples

and plots:
- CPU (NumPy) vs GPU Basic vs GPU Optimized vs GPU Fast RNG

The key result is that:
- naive GPU approaches can underperform due to CPU bottlenecks
- **GPU-side RNG is required for real speed-up**

---

##  Future Improvements

- Use per-thread RNG states for improved statistical independence
- Explore warp-level primitives for faster reductions
- Add CLI runner (`python run_benchmarks.py`)
- Compare against CuPy / CUDA C implementation

---

##  Author

**Aryan Sikhwal**  
GitHub: https://github.com/aryansikhwal  
LinkedIn: www.linkedin.com/in/aryansikhwal

