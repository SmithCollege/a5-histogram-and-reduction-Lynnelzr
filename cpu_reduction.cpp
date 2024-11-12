#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

const long long MOD = 1000000007;

int cpuReductionSum(const std::vector<int>& data) {
    return std::accumulate(data.begin(), data.end(), 0);
}

long long cpuReductionProduct(const std::vector<int>& data) {
    bool all_zeros = true;
    long long product = std::accumulate(data.begin(), data.end(), 1LL,
        [&all_zeros](long long total, int value) {
            if (value != 0) {
                all_zeros = false;
                total = (total * value) % MOD;
            }
            return total;
        });
    return all_zeros ? 0 : product;
}

int cpuReductionMin(const std::vector<int>& data) {
    return *std::min_element(data.begin(), data.end());
}

int cpuReductionMax(const std::vector<int>& data) {
    return *std::max_element(data.begin(), data.end());
}

int main() {
    const int maxSize = 16384;  // Maximum size of the array
    const int initialSize = 1024; // Starting size of the array
    const int factor = 2;         // Factor by which array size will increase

    for (int size = initialSize; size <= maxSize; size *= factor) {
        // Generate test data
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 1); // Fill with sequential numbers starting from 1

        std::cout << "Array Size: " << size << std::endl;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        int sum = cpuReductionSum(data);
        long long product = cpuReductionProduct(data);
        int min = cpuReductionMin(data);
        int max = cpuReductionMax(data);

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "  Sum: " << sum << std::endl;
        std::cout << "  Product (mod " << MOD << "): " << product << std::endl;
        std::cout << "  Min: " << min << std::endl;
        std::cout << "  Max: " << max << std::endl;
        std::cout << "  Runtime: " << elapsed.count() << " milliseconds" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}

