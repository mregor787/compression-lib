#include "bwt.h"
#include "rle.h"
#include "huffman.h"
#include "compression.h"

void compress_bwt_rle_huffman(const char* input_file, const char* output_file, ExecutionMode mode) {
    const char* temp_bwt = "temp_bwt.bin";
    const char* temp_rle = "temp_rle.bin";

    clock_t global_start = clock();

    if (mode == MODE_CPU) {
        printf("[MODE] CPU: BWT -> RLE -> Huffman\n");
        bwt_transform_cpu(input_file, temp_bwt);
        rle_compress_cpu(temp_bwt, temp_rle);
    } else if (mode == MODE_CUDA) {
        printf("[MODE] CUDA: BWT (GPU) -> RLE (GPU) -> Huffman (CPU)\n");
        bwt_transform_cuda(input_file, temp_bwt);
        rle_compress_cuda(temp_bwt, temp_rle);
    } else {
        fprintf(stderr, "Unknown execution mode\n");
        return;
    }

    huffman_compress_cpu(temp_rle, output_file);

    clock_t global_end = clock();
    double elapsed_ms = 1000.0 * (global_end - global_start) / CLOCKS_PER_SEC;
    printf("%s BWT+RLE+Huffman %s total time: %.2f ms\n", input_file,
           mode == MODE_CPU ? "CPU" : "CUDA", elapsed_ms);

    remove(temp_bwt);
    remove(temp_rle);
}

void decompress_bwt_rle_huffman(const char* input_file, const char* output_file, ExecutionMode mode) {
    const char* temp_rle = "temp_rle_d.bin";
    const char* temp_bwt = "temp_bwt_d.bin";

    clock_t global_start = clock();

    printf("[MODE] %s: Huffman -> RLE -> BWT\n", mode == MODE_CPU ? "CPU" : "CUDA");

    huffman_decompress_cpu(input_file, temp_rle);

    if (mode == MODE_CPU) {
        rle_decompress_cpu(temp_rle, temp_bwt);
        bwt_inverse_cpu(temp_bwt, output_file);
    } else if (mode == MODE_CUDA) {
        rle_decompress_cuda(temp_rle, temp_bwt);
        bwt_inverse_cuda(temp_bwt, output_file);
    } else {
        fprintf(stderr, "Unknown execution mode\n");
        return;
    }

    clock_t global_end = clock();
    double elapsed_ms = 1000.0 * (global_end - global_start) / CLOCKS_PER_SEC;
    printf("%s Huffman+RLE+BWT %s total time: %.2f ms\n", input_file,
           mode == MODE_CPU ? "CPU" : "CUDA", elapsed_ms);

    remove(temp_rle);
    remove(temp_bwt);
}
