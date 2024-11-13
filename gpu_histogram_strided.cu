#include <iostream>
#include <cuda_runtime.h>

__global__ void gpuHistogramStrided(int* input, int* histogram, int n, int numBins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        int bin = input[i] % numBins;
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    int numBins = 10;
    int *d_data, *d_histogram;

    // Loop over different array sizes (powers of 2)
    for (int arraySize = 1024; arraySize <= 1048576; arraySize *= 2) {
        // Allocate host memory
        int* h_data = new int[arraySize];
        int* h_histogram = new int[numBins]();

        // Initialize data on the host
        for (int i = 0; i < arraySize; ++i) {
            h_data[i] = i % numBins;
        }

        // Allocate device memory
        cudaMalloc(&d_data, arraySize * sizeof(int));
        cudaMalloc(&d_histogram, numBins * sizeof(int));

        // Copy data to device and initialize histogram on device
        cudaMemcpy(d_data, h_data, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_histogram, 0, numBins * sizeof(int));

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start timing
        cudaEventRecord(start, 0);

        // Launch kernel
        gpuHistogramStrided<<<(arraySize + 255) / 256, 256>>>(d_data, d_histogram, arraySize, numBins);

        // Stop timing
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy result back to host
        cudaMemcpy(h_histogram, d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);

        // Print results for this array size
        std::cout << "Array Size: " << arraySize << "\nHistogram:\n";
        for (int i = 0; i < numBins; ++i) {
            std::cout << "Bin " << i << ": " << h_histogram[i] << std::endl;
        }
        std::cout << "GPU Runtime: " << milliseconds << " ms\n";
        std::cout << "----------------------------------------\n";

        // Free allocated memory
        delete[] h_data;
        delete[] h_histogram;
        cudaFree(d_data);
        cudaFree(d_histogram);

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}

