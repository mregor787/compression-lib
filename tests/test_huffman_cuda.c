#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../include/huffman.h"

#define MAX_SIZE (1024 * 1024)

int main() {
    const char *input_str = "THIS IS A TEST STRING FOR HUFFMAN CUDA";
    size_t input_size = strlen(input_str);

    uint8_t *input = malloc(input_size);
    memcpy(input, input_str, input_size);

    uint8_t *compressed   = malloc(MAX_SIZE);
    uint8_t *decompressed = malloc(MAX_SIZE);
    size_t compressed_size = 0;
    size_t decompressed_size = 0;

    printf("[Huffman CUDA Test]\n");
    printf("Input size:        %zu bytes\n", input_size);

    int comp_res = huffman_compress_cuda(input, input_size, compressed, &compressed_size);
    if (comp_res != 0) {
        printf("Compression failed (code %d)\n", comp_res);
        free(input); free(compressed); free(decompressed);
        return 1;
    }

    printf("Compressed size:   %zu bytes\n", compressed_size);

    int decomp_res = huffman_decompress_cpu(compressed, compressed_size, decompressed, &decompressed_size);
    if (decomp_res != 0) {
        printf("Decompression failed (code %d)\n", decomp_res);
        free(input); free(compressed); free(decompressed);
        return 1;
    }

    printf("Decompressed size: %zu bytes\n", decompressed_size);
    printf("Decompressed:      \"%.*s\"\n", (int)decompressed_size, decompressed);

    if (decompressed_size != input_size || memcmp(input, decompressed, input_size) != 0) {
        for (size_t i = 0; i < input_size && i < decompressed_size; ++i) {
            if (input[i] != decompressed[i]) {
                printf("Mismatch at byte %zu: original '%c' (0x%02x), decompressed '%c' (0x%02x)\n",
                       i, input[i], input[i], decompressed[i], decompressed[i]);
                break;
            }
        }

        if (decompressed_size != input_size)
            printf("Different sizes: input = %zu, decompressed = %zu\n", input_size, decompressed_size);

        printf("Mismatch after decompression!\n");
        free(input); free(compressed); free(decompressed);
        return 1;
    }

    printf("Huffman CUDA compression/decompression successful.\n");
    free(input); free(compressed); free(decompressed);
    return 0;
}
