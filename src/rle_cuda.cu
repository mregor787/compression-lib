#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/rle.h"

__global__ void prefix_sum(const int *input, int *output, int n, bool exclusive) {
    __shared__ int temp[1024];
    int tid = threadIdx.x;

    temp[tid] = (tid < n) ? input[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        int t = temp[tid];
        if (tid >= offset)
            t += temp[tid - offset];
        __syncthreads();
        temp[tid] = t;
        __syncthreads();
    }

    if (tid < n) {
        if (exclusive)
            output[tid] = (tid == 0) ? 0 : temp[tid - 1];
        else
            output[tid] = temp[tid];
    }
}

__global__ void mark_series(const uint8_t *input, int *flags, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0 || input[i] != input[i - 1])
        flags[i] = 1;
    else
        flags[i] = 0;
}

__global__ void write_compressed(const uint8_t *input, size_t n,
                          const int *flags, const int *positions,
                          uint8_t *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (flags[i] == 1) {
        int start = i;
        uint8_t byte = input[i];
        int len = 1;
        while (start + len < n && input[start + len] == byte)
            len++;

        int out_pos = positions[i] * 2;
        output[out_pos] = byte;
        output[out_pos + 1] = (uint8_t)(len);
    }
}

__global__ void write_decompressed(const uint8_t *input,
                                   const int *positions,
                                   uint8_t *output, int num_pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;

    uint8_t value = input[2 * i];
    int count = input[2 * i + 1];
    int start = positions[i];

    for (int j = 0; j < count; ++j) {
        output[start + j] = value;
    }
}

int rle_compress_cuda(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size == 0)
        return -1;

    uint8_t *d_input, *d_output;
    int *d_flags, *d_positions;
    size_t max_output_size = input_size * 2;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, max_output_size);
    cudaMalloc((void**)&d_flags, input_size * sizeof(int));
    cudaMalloc((void**)&d_positions, input_size * sizeof(int));
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (input_size + threads - 1) / threads;

    mark_series<<<blocks, threads>>>(d_input, d_flags, input_size);
    cudaDeviceSynchronize();

    prefix_sum<<<1, 1024>>>(d_flags, d_positions, input_size, false);
    cudaDeviceSynchronize();

    write_compressed<<<blocks, threads>>>(d_input, input_size, d_flags, d_positions, d_output);
    cudaDeviceSynchronize();

    int last_pos = 0;
    cudaMemcpy(&last_pos, &d_positions[input_size - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int num_series = last_pos + 1;

    cudaMemcpy(output, d_output, num_series * 2, cudaMemcpyDeviceToHost);
    *output_size = num_series * 2;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    cudaFree(d_positions);

    return 0;
}

int rle_decompress_cuda(const uint8_t *input, size_t input_size,
                        uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size % 2 != 0)
        return -1;

    int num_pairs = input_size / 2;

    uint8_t *d_input, *d_output;
    int *d_counts, *d_positions;

    cudaMalloc((void**)&d_input, input_size);
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_counts, num_pairs * sizeof(int));
    cudaMalloc((void**)&d_positions, num_pairs * sizeof(int));

    // Извлекаем длины серий
    int *h_counts = (int*)malloc(num_pairs * sizeof(int));
    for (int i = 0; i < num_pairs; ++i)
        h_counts[i] = input[2 * i + 1];
    cudaMemcpy(d_counts, h_counts, num_pairs * sizeof(int), cudaMemcpyHostToDevice);
    free(h_counts);

    // Префикс-сумма по длинам
    prefix_sum<<<1, 1024>>>(d_counts, d_positions, num_pairs, true);
    cudaDeviceSynchronize();

    int last_pos = 0, last_len = input[input_size - 1];
    cudaMemcpy(&last_pos, &d_positions[num_pairs - 1], sizeof(int), cudaMemcpyDeviceToHost);
    *output_size = last_pos + last_len;

    cudaMalloc((void**)&d_output, *output_size);

    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;

    write_decompressed<<<blocks, threads>>>(d_input, d_positions, d_output, num_pairs);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, *output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counts);
    cudaFree(d_positions);

    return 0;
}
