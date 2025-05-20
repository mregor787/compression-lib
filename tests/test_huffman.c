#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../include/huffman.h"

#define MAX_SIZE 2048

int main() {
    printf("[Huffman Test]\n");

    const char *text = "AAAAABBBBBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEE";
    size_t input_size = strlen(text);

    uint8_t *input = (uint8_t *)text;
    uint8_t compressed[MAX_SIZE];
    uint8_t decompressed[MAX_SIZE];
    size_t compressed_size = 0;
    size_t decompressed_size = MAX_SIZE;

    int res1 = huffman_compress_cpu(input, input_size, compressed, &compressed_size);
    int res2 = huffman_decompress_cpu(compressed, compressed_size, decompressed, &decompressed_size);

    printf("Input size:        %zu bytes\n", input_size);
    printf("Compressed size:   %zu bytes\n", compressed_size);
    printf("Decompressed size: %zu bytes\n", decompressed_size);

    if (res1 == 0 && res2 == 0 &&
        memcmp(input, decompressed, input_size) == 0) {
        printf("Huffman compression/decompression successful.\n");
        return 0;
    } else {
        printf("Mismatch after decompression!\n");
        return 1;
    }
}
