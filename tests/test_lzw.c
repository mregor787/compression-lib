#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../include/lzw.h"

#define MAX_BUFFER_SIZE 2048

int main() {
    const char *input_str = "TOBEORNOTTOBEORTOBEORNOT";
    size_t input_size = strlen(input_str);

    uint8_t compressed[MAX_BUFFER_SIZE];
    uint8_t decompressed[MAX_BUFFER_SIZE];
    size_t compressed_size = 0, decompressed_size = 0;

    printf("[LZW Test]\n");
    printf("Input size:        %zu bytes\n", input_size);

    if (lzw_compress_cpu((const uint8_t *)input_str, input_size,
                         compressed, &compressed_size) != 0) {
        printf("Compression failed!\n");
        return 1;
    }

    printf("Compressed size:   %zu bytes\n", compressed_size);

    if (lzw_decompress_cpu(compressed, compressed_size,
                           decompressed, &decompressed_size) != 0) {
        printf("Decompression failed!\n");
        return 1;
    }

    printf("Decompressed size: %zu bytes\n", decompressed_size);

    if (input_size != decompressed_size ||
        memcmp(input_str, decompressed, input_size) != 0) {
        printf("Mismatch after decompression!\n");
        return 1;
    }

    printf("LZW compression/decompression successful.\n");
    return 0;
}
