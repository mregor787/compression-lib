#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include "rle.h"

void rle_compress_cuda(const char* input_file, const char* output_file)
{
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Cannot open file: %s\n", input_file);
        return;
    }

    fseek(in, 0, SEEK_END);
    size_t filesize = ftell(in);
    fseek(in, 0, SEEK_SET);

    uint8_t* h_input = (uint8_t*)malloc(filesize);
    fread(h_input, 1, filesize, in);
    fclose(in);

    thrust::device_vector<uint8_t> input_data(h_input, h_input + filesize);
    thrust::device_vector<uint32_t> mask(filesize, 0);

    clock_t start = clock();

    thrust::transform(
        input_data.begin() + 1,
        input_data.end(),
        input_data.begin(),
        mask.begin() + 1,
        thrust::not_equal_to<uint8_t>()
    );
    mask[0] = 1;

    thrust::device_vector<uint32_t> scanned_mask(filesize);
    thrust::inclusive_scan(mask.begin(), mask.end(), scanned_mask.begin());

    thrust::device_vector<uint32_t> indices(filesize);
    thrust::sequence(indices.begin(), indices.end());

    uint32_t total_size = thrust::reduce(mask.begin(), mask.end());
    thrust::device_vector<uint32_t> compacted_mask(total_size);
    thrust::copy_if(
        indices.begin(), indices.end(),
        mask.begin(),
        compacted_mask.begin(),
        cuda::std::identity{}
    );

    thrust::device_vector<uint8_t> output_data(total_size);
    thrust::device_vector<uint32_t> occurences(total_size);

    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(total_size),
        output_data.begin(),
        [in_ptr = thrust::raw_pointer_cast(input_data.data()),
         comp_ptr = thrust::raw_pointer_cast(compacted_mask.data())] __device__ (uint32_t i)
        {
            return in_ptr[comp_ptr[i]];
        }
    );

    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(total_size),
        occurences.begin(),
        [comp_ptr = thrust::raw_pointer_cast(compacted_mask.data()),
         n = static_cast<uint32_t>(filesize), total_size] __device__ (uint32_t i)
        {
            uint32_t start = comp_ptr[i];
            uint32_t end = (i + 1 < total_size) ? comp_ptr[i + 1] : n;
            return end - start;
        }
    );

    cudaDeviceSynchronize();

    clock_t end = clock();
    printf("%s RLE GPU compression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        free(h_input);
        return;
    }

    fwrite(&total_size, sizeof(uint32_t), 1, out);
    thrust::host_vector<uint8_t> h_output = output_data;
    thrust::host_vector<uint32_t> h_counts = occurences;
    fwrite(h_output.data(), sizeof(uint8_t), total_size, out);
    fwrite(h_counts.data(), sizeof(uint32_t), total_size, out);
    fclose(out);

    free(h_input);
}

__global__ void scatter_decompress_kernel(
    const uint8_t* input,
    const uint32_t* occurences,
    const uint32_t* positions,
    uint8_t* output,
    uint32_t totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < totalSize; i += stride) {
        uint8_t value = input[i];
        uint32_t start = positions[i];
        uint32_t count = occurences[i];

        for (uint32_t j = 0; j < count; ++j) {
            output[start + j] = value;
        }
    }
}

void rle_decompress_cuda(const char* input_file, const char* output_file)
{
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Cannot open file: %s\n", input_file);
        return;
    }

    uint32_t total_size = 0;
    fread(&total_size, sizeof(uint32_t), 1, in);

    uint8_t* h_data = (uint8_t*)malloc(total_size);
    uint32_t* h_counts = (uint32_t*)malloc(total_size * sizeof(uint32_t));
    fread(h_data, sizeof(uint8_t), total_size, in);
    fread(h_counts, sizeof(uint32_t), total_size, in);
    fclose(in);

    thrust::device_vector<uint8_t> compressed(h_data, h_data + total_size);
    thrust::device_vector<uint32_t> occurences(h_counts, h_counts + total_size);
    thrust::device_vector<uint32_t> positions(total_size);

    clock_t start = clock();

    thrust::exclusive_scan(occurences.begin(), occurences.end(), positions.begin());

    uint32_t decompressed_size = positions[total_size - 1] + h_counts[total_size - 1];
    thrust::device_vector<uint8_t> output_data(decompressed_size);

    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    scatter_decompress_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(compressed.data()),
        thrust::raw_pointer_cast(occurences.data()),
        thrust::raw_pointer_cast(positions.data()),
        thrust::raw_pointer_cast(output_data.data()),
        total_size
    );
    cudaDeviceSynchronize();

    clock_t end = clock();
    printf("%s GPU decompression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        free(h_data); free(h_counts);
        return;
    }

    thrust::host_vector<uint8_t> h_output = output_data;
    fwrite(h_output.data(), sizeof(uint8_t), decompressed_size, out);
    fclose(out);

    free(h_data);
    free(h_counts);
}
