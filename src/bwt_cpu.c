#include "bwt.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t index;
    const uint8_t *text;
    size_t size;
} Rotation;

static int compare_rotations(const void *a, const void *b) {
    const Rotation *ra = (const Rotation *)a;
    const Rotation *rb = (const Rotation *)b;

    for (size_t i = 0; i < ra->size; ++i) {
        uint8_t ca = ra->text[(ra->index + i) % ra->size];
        uint8_t cb = rb->text[(rb->index + i) % rb->size];
        if (ca != cb)
            return (int)ca - (int)cb;
    }
    return 0;
}

int bwt_transform_cpu(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *primary_index) {
    if (!input || !output || !primary_index || input_size == 0)
        return -1;

    Rotation *rotations = malloc(input_size * sizeof(Rotation));
    if (!rotations) return -2;

    for (size_t i = 0; i < input_size; ++i) {
        rotations[i].index = i;
        rotations[i].text = input;
        rotations[i].size = input_size;
    }

    qsort(rotations, input_size, sizeof(Rotation), compare_rotations);

    for (size_t i = 0; i < input_size; ++i) {
        size_t idx = (rotations[i].index + input_size - 1) % input_size;
        output[i] = input[idx];
        if (rotations[i].index == 0)
            *primary_index = i;
    }

    free(rotations);
    return 0;
}

int bwt_inverse_cpu(const uint8_t *input, size_t input_size,
                    uint8_t *output, size_t primary_index) {
    if (!input || !output || input_size == 0 || primary_index >= input_size)
        return -1;

    int counts[256] = {0};
    int totals[256] = {0};
    int *next = malloc(input_size * sizeof(int));
    if (!next) return -2;

    for (size_t i = 0; i < input_size; ++i)
        counts[input[i]]++;

    int sum = 0;
    for (int i = 0; i < 256; ++i) {
        totals[i] = sum;
        sum += counts[i];
    }

    for (size_t i = 0; i < input_size; ++i)
        next[i] = totals[input[i]]++;

    size_t pos = primary_index;
    for (size_t i = 0; i < input_size; ++i) {
        output[input_size - i - 1] = input[pos];
        pos = next[pos];
    }

    free(next);
    return 0;
}
