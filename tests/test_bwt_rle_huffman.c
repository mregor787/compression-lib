#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../include/bwt_rle_huffman.h"

#define MAX_SIZE 2048

int main() {
    printf("[BWT + RLE + Huffman Test]\n");

    const char *text = "AAAAABBBBBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEE";
    size_t input_size = strlen(text);

    uint8_t *input = (uint8_t *)text;
    uint8_t compressed[MAX_SIZE];
    uint8_t decompressed[MAX_SIZE];
    size_t compressed_size = 0;
    size_t decompressed_size = MAX_SIZE;

    int res1 = compress_bwt_rle_huffman(input, input_size, compressed, &compressed_size, MODE_CPU);
    int res2 = decompress_bwt_rle_huffman(compressed, compressed_size, decompressed, &decompressed_size, MODE_CPU);

    printf("Input size:        %zu bytes\n", input_size);
    printf("Compressed size:   %zu bytes\n", compressed_size);
    printf("Decompressed size: %zu bytes\n", decompressed_size);
    printf("Decompressed:      \"%.*s\"\n", (int)decompressed_size, decompressed);

    if (res1 == 0 && res2 == 0 && input_size == decompressed_size &&
        memcmp(input, decompressed, input_size) == 0) {
        printf("BWT+RLE+Huffman compression/decompression successful.\n");
        return 0;
    } else {
        printf("Mismatch after decompression!\n");
        return 1;
    }
}
