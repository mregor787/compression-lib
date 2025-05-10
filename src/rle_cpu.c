#include "rle.h"

int rle_compress_cpu(const uint8_t *input, size_t input_size,
                     uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size) return -1;

    size_t in_pos = 0, out_pos = 0;
    while (in_pos < input_size) {
        uint8_t current = input[in_pos];
        size_t run_length = 1;

        while (in_pos + run_length < input_size && 
               input[in_pos + run_length] == current && 
               run_length < 255) {
            run_length++;
        }

        output[out_pos++] = current;
        output[out_pos++] = run_length;

        in_pos += run_length;
    }

    *output_size = out_pos;
    return 0;
}

int rle_decompress_cpu(const uint8_t *input, size_t input_size,
                       uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size) return -1;

    size_t in_pos = 0, out_pos = 0;
    while (in_pos + 1 < input_size) {
        uint8_t value = input[in_pos++];
        uint8_t count = input[in_pos++];

        for (int i = 0; i < count; ++i) {
            output[out_pos++] = value;
        }
    }

    *output_size = out_pos;
    return 0;
}
