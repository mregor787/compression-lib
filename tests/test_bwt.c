#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../include/bwt.h"

#define MAX_SIZE 1024

int main() {
    printf("[BWT Test]\n");

    const char *text = "banana_bandana";
    size_t input_size = strlen(text);

    uint8_t *input = (uint8_t *)text;
    uint8_t transformed[MAX_SIZE];
    uint8_t restored[MAX_SIZE];
    size_t primary_index = 0;

    int res1 = bwt_transform_cpu(input, input_size, transformed, &primary_index);
    int res2 = bwt_inverse_cpu(transformed, input_size, restored, primary_index);

    printf("Input size:        %zu bytes\n", input_size);
    printf("Primary index:     %zu\n", primary_index);

    if (res1 == 0 && res2 == 0 && memcmp(input, restored, input_size) == 0) {
        printf("BWT transform/inverse successful.\n");
        return 0;
    } else {
        printf("Mismatch after inverse BWT!\n");
        return 1;
    }
}
