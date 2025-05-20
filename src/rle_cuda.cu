#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/rle.h"

__global__ void mark_series(const uint8_t *input, int *flags, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || input[i] != input[i - 1]) ? 1 : 0;
}

__global__ void write_compressed(const uint8_t *input, size_t n,
                                 const int *flags, const int *positions,
                                 uint8_t *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || flags[i] == 0) return;

    int start = i;
    uint8_t byte = input[i];
    int len = 1;
    while (start + len < n && input[start + len] == byte)
        len++;

    int out_pos = positions[i] * 2;
    output[out_pos]     = byte;
    output[out_pos + 1] = (uint8_t)(len);
}

__global__ void write_decompressed(const uint8_t *input,
                                   const int *positions,
                                   uint8_t *output, int num_pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;

    uint8_t value = input[2 * i];
    uint8_t count = input[2 * i + 1];
    int start = positions[i];

    for (uint8_t j = 0; j < count; ++j) {
        output[start + j] = value;
    }
}

int rle_compress_cuda(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size == 0)
        return -1;

    thrust::device_vector<uint8_t> d_input(input, input + input_size);
    thrust::device_vector<int> d_flags(input_size);
    thrust::device_vector<int> d_positions(input_size);

    int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    mark_series<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                     thrust::raw_pointer_cast(d_flags.data()),
                                     input_size);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_flags.begin(), d_flags.end(), d_positions.begin());

    int last_flag = 0, last_pos = 0;
    cudaMemcpy(&last_flag, thrust::raw_pointer_cast(d_flags.data() + input_size - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_pos, thrust::raw_pointer_cast(d_positions.data() + input_size - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int num_series = last_flag ? last_pos + 1 : last_pos;

    thrust::device_vector<uint8_t> d_output(num_series * 2);

    write_compressed<<<blocks, threads>>>(thrust::raw_pointer_cast(d_input.data()),
                                          input_size,
                                          thrust::raw_pointer_cast(d_flags.data()),
                                          thrust::raw_pointer_cast(d_positions.data()),
                                          thrust::raw_pointer_cast(d_output.data()));
    cudaDeviceSynchronize();

    cudaMemcpy(output, thrust::raw_pointer_cast(d_output.data()), num_series * 2, cudaMemcpyDeviceToHost);
    *output_size = num_series * 2;

    return 0;
}

int rle_decompress_cuda(const uint8_t *input, size_t input_size,
                        uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size % 2 != 0)
        return -1;

    int num_pairs = input_size / 2;

    thrust::device_vector<uint8_t> d_input(input, input + input_size);
    thrust::device_vector<int> d_counts(num_pairs);

    for (int i = 0; i < num_pairs; ++i)
        d_counts[i] = input[i * 2 + 1];

    thrust::device_vector<int> d_positions(num_pairs);
    thrust::exclusive_scan(thrust::device, d_counts.begin(), d_counts.end(), d_positions.begin());

    int total_output_size = 0;
    cudaMemcpy(&total_output_size, thrust::raw_pointer_cast(d_positions.data() + num_pairs - 1),
               sizeof(int), cudaMemcpyDeviceToHost);
    total_output_size += input[input_size - 1];

    thrust::device_vector<uint8_t> d_output(total_output_size);

    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;
    write_decompressed<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_positions.data()),
        thrust::raw_pointer_cast(d_output.data()),
        num_pairs
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, thrust::raw_pointer_cast(d_output.data()), total_output_size, cudaMemcpyDeviceToHost);
    *output_size = total_output_size;

    return 0;
}
