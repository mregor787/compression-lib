#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../include/bwt.h"

#define MAX_SIZE 256

int main() {
    const char *input_str = "banana_bandana";
    size_t len = strlen(input_str);

    uint8_t input[MAX_SIZE];
    uint8_t bwt_out[MAX_SIZE];
    uint8_t restored[MAX_SIZE];
    size_t primary_index = -1;

    memcpy(input, input_str, len);

    printf("[BWT CUDA Test]\n");
    printf("Input string:       \"%s\"\n", input_str);

    if (bwt_transform_cuda(input, len, bwt_out, &primary_index) != 0) {
        fprintf(stderr, "CUDA BWT transform failed\n");
        return 1;
    }

    printf("BWT output:         \"");
    for (size_t i = 0; i < len; ++i) putchar(bwt_out[i]);
    printf("\"\n");
    printf("Primary index:      %ld\n", primary_index);

    if (bwt_inverse_cuda(bwt_out, len, restored, primary_index) != 0) {
        fprintf(stderr, "CUDA BWT inverse failed\n");
        return 1;
    }

    printf("Restored string:    \"");
    for (size_t i = 0; i < len; ++i) putchar(restored[i]);
    printf("\"\n");

    if (memcmp(input, restored, len) != 0) {
        fprintf(stderr, "Mismatch after BWT roundtrip!\n");
        return 1;
    }

    printf("CUDA BWT transform/inverse successful.\n");
    return 0;
}
