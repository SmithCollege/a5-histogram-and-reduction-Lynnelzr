#include <iostream>
#include <cuda_runtime.h>

__global__ void gpuReductionSum(int* input, int* output, int n) {
    extern __shared__ int shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared_data[tid] = (i < n ? input[i] : 0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared_data[0];
}

int main() {
    int initial_size = 1024;
    int max_size = 1048576; // 1 million elements for example

    for (int n = initial_size; n <= max_size; n *= 2) {
        // Allocate and initialize host data
        int *h_data = new int[n];
        int *h_result = new int[(n + 1023) / 1024];

        // Fill data with a simple pattern, e.g., numbers from 1 to n
        for (int i = 0; i < n; ++i) h_data[i] = 1; // Set all to 1 to make sum predictable

        // Allocate device memory
        int *d_data, *d_result;
        cudaMalloc(&d_data, n * sizeof(int));
        cudaMalloc(&d_result, (n + 1023) / 1024 * sizeof(int));

        // Copy data from host to device
        cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

        // Launch the kernel with appropriate grid and block sizes
        int numBlocks = (n + 1023) / 1024;

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start timing
        cudaEventRecord(start, 0);
        
        gpuReductionSum<<<numBlocks, 512, 512 * sizeof(int)>>>(d_data, d_result, n);

        // Stop timing
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy the result back to the host
        cudaMemcpy(h_result, d_result, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        // Perform final reduction on the host to get the total sum
        int sum = 0;
        for (int i = 0; i < numBlocks; ++i) {
            sum += h_result[i];
        }

        // Print the result for the current array size and time
        std::cout << "Array Size: " << n << " | Sum: " << sum << " | GPU Runtime: " << milliseconds << " ms" << std::endl;

        // Free host and device memory for this iteration
        delete[] h_data;
        delete[] h_result;
        cudaFree(d_data);
        cudaFree(d_result);

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}

