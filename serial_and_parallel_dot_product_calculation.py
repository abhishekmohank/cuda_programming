import time
import matplotlib.pyplot as plt

# Serial version
def dot_product_serial(a, b):
    start_time = time.time()
    result = sum(x * y for x, y in zip(a, b))
    end_time = time.time()
    return result, end_time - start_time

# Parallel version
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void dotProduct(int *a, int *b, int *result, int n) {
    __shared__ int partialSum[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int partial = 0;

    if (tid < n) {
        partial = a[tid] * b[tid];
    }

    partialSum[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = partialSum[0];
    }
}
""")

def dot_product_parallel(a, b):
    start_time = time.time()
    n = len(a)
    h_a = np.array(a).astype(np.int32)
    h_b = np.array(b).astype(np.int32)

    d_a = cuda.mem_alloc(h_a.nbytes)
    d_b = cuda.mem_alloc(h_b.nbytes)

    cuda.memcpy_htod(d_a, h_a)
    cuda.memcpy_htod(d_b, h_b)

    blockSize = 256
    gridSize = (n + blockSize - 1) // blockSize

    d_partial_products = cuda.mem_alloc(gridSize * np.dtype(np.int32).itemsize)

    func = mod.get_function("dotProduct")
    func(d_a, d_b, d_partial_products, np.int32(n), block=(blockSize, 1, 1), grid=(gridSize, 1))

    h_partial_products = np.empty(gridSize, dtype=np.int32)
    cuda.memcpy_dtoh(h_partial_products, d_partial_products)

    dot_product = np.sum(h_partial_products)

    d_a.free()
    d_b.free()
    d_partial_products.free()

    end_time = time.time()
    return dot_product, end_time - start_time

# Generate input vectors
vector_sizes = [10**i for i in range(1, 7)]  # Vary the size of vectors from 10 to 1,000,000
serial_times = []
parallel_times = []

# Measure execution time for different vector sizes
for size in vector_sizes:
    a = list(range(size))
    b = [2 * x for x in a]

    serial_result, serial_time = dot_product_serial(a, b)
    parallel_result, parallel_time = dot_product_parallel(a, b)

    serial_times.append(serial_time)
    parallel_times.append(parallel_time)

    print(f"Vector Size: {size}")
    print(f"Serial Dot Product: {serial_result}, Execution Time: {serial_time:.6f} s")
    print(f"Parallel Dot Product: {parallel_result}, Execution Time: {parallel_time:.6f} s")
    print()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(vector_sizes, serial_times, label='Serial')
plt.plot(vector_sizes, parallel_times, label='Parallel')
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Comparison: Serial vs. Parallel Dot Product')
plt.xscale('log')  # Logarithmic scale for x-axis
plt.yscale('log')  # Logarithmic scale for y-axis
plt.legend()
plt.grid(True)
plt.show()
