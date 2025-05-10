#include "lz77.h"
#include <string.h>

#define WINDOW_SIZE 4096     // Размер скользящего окна
#define LOOKAHEAD_SIZE 15    // Максимальный размер совпадения

int lz77_compress_cpu(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size)
        return -1;

    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos < input_size) {
        size_t best_offset = 0;
        size_t best_length = 0;

        size_t window_start = (in_pos >= WINDOW_SIZE) ? (in_pos - WINDOW_SIZE) : 0;

        for (size_t j = window_start; j < in_pos; ++j) {
            size_t length = 0;
            while (length < LOOKAHEAD_SIZE &&
                j + length < in_pos &&
                in_pos + length < input_size &&
                input[j + length] == input[in_pos + length]) {
                ++length;
            }

            if (length > best_length) {
                best_length = length;
                best_offset = in_pos - j;
            }
        }

        if (in_pos + best_length >= input_size) {
            best_length = 0;
            best_offset = 0;
        }

        uint8_t next_symbol = input[in_pos + best_length];

        output[out_pos++] = (best_offset >> 8) & 0xFF;
        output[out_pos++] = best_offset & 0xFF;
        output[out_pos++] = (uint8_t)best_length;
        output[out_pos++] = next_symbol;

        in_pos += best_length + 1;
    }


    *output_size = out_pos;
    return 0;
}

int lz77_decompress_cpu(const uint8_t *input, size_t input_size,
                        uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size)
        return -1;

    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos + 4 <= input_size) {
        uint16_t offset = ((uint16_t)input[in_pos] << 8) | input[in_pos + 1];
        uint8_t length = input[in_pos + 2];
        uint8_t symbol = input[in_pos + 3];
        in_pos += 4;

        if (offset > 0) {
            if (out_pos < offset) return -2; // выход за границу
            for (size_t i = 0; i < length; ++i) {
                output[out_pos] = output[out_pos - offset];
                ++out_pos;
            }
        }

        output[out_pos++] = symbol;
    }

    *output_size = out_pos;
    return 0;
}
