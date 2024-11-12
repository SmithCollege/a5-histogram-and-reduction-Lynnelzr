#include <iostream>
#include <vector>
#include <chrono>

// Function to calculate histogram on the CPU
void cpuHistogram(const std::vector<int>& data, int numBins, std::vector<int>& histogram) {
    for (int val : data) {
        histogram[val % numBins]++;
    }
}

int main() {
    int initialSize = 1024;        // Start with 1024 elements
    int maxSize = 1048576;         // Maximum array size, e.g., 1,048,576 elements
    int numBins = 10;              // Number of bins for the histogram

    for (int size = initialSize; size <= maxSize; size *= 2) {
        std::vector<int> data(size);
        std::vector<int> histogram(numBins, 0);

        // Fill the data array with some values (for simplicity, values modulo numBins)
        for (int i = 0; i < size; ++i) {
            data[i] = i % numBins;
        }

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Calculate histogram
        cpuHistogram(data, numBins, histogram);

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        // Display results
        std::cout << "Array Size: " << size << "\nHistogram:\n";
        for (int i = 0; i < numBins; ++i) {
            std::cout << "Bin " << i << ": " << histogram[i] << std::endl;
        }
        std::cout << "Runtime: " << elapsed.count() << " ms\n";
        std::cout << "----------------------------------------\n";
    }

    return 0;
}

