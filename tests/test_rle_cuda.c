#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../include/rle.h"

#define MAX_SIZE (1024 * 1024)

int main() {
    const char *input_str = "AAAAABBBBBCCCCDDDEEFFGGG";
    size_t input_size = strlen(input_str);

    uint8_t *input        = malloc(input_size);
    uint8_t *compressed   = malloc(MAX_SIZE);
    uint8_t *decompressed = malloc(MAX_SIZE);
    memcpy(input, input_str, input_size);

    size_t compressed_size = 0;
    size_t decompressed_size = 0;

    printf("[RLE CUDA Test]\n");
    printf("Input size:        %zu bytes\n", input_size);

    int comp_res = rle_compress_cuda(input, input_size, compressed, &compressed_size);
    if (comp_res != 0) {
        printf("Compression failed (code %d)\n", comp_res);
        return 1;
    }

    printf("Compressed size:   %zu bytes\n", compressed_size);

    int decomp_res = rle_decompress_cuda(compressed, compressed_size, decompressed, &decompressed_size);
    if (decomp_res != 0) {
        printf("Decompression failed (code %d)\n", decomp_res);
        return 1;
    }

    printf("Decompressed size: %zu bytes\n", decompressed_size);
    printf("Decompressed:      \"%.*s\"\n", (int)decompressed_size, decompressed);

    if (decompressed_size != input_size || memcmp(input, decompressed, input_size) != 0) {
        printf("Mismatch after decompression!\n");
        return 1;
    }

    printf("RLE CUDA compression/decompression successful.\n");

    free(input);
    free(compressed);
    free(decompressed);
    return 0;
}
