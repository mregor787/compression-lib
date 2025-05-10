#include "bwt.h"
#include "rle.h"
#include "huffman.h"
#include "compression.h"
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE (10 * 1024 * 1024)

int compress_bwt_rle_huffman(const uint8_t *input, size_t input_size,
                              uint8_t *output, size_t *output_size,
                              ExecutionMode mode) {
    if (!input || !output || !output_size) return -1;

    // Этап 1: BWT
    uint8_t *bwt_out = (uint8_t *)malloc(input_size);
    if (!bwt_out) return -10;

    size_t primary_index = 0;
    int bwt_res = (mode == MODE_CUDA)
        ? bwt_transform_cuda(input, input_size, bwt_out, &primary_index)
        : bwt_transform_cpu(input, input_size, bwt_out, &primary_index);

    if (bwt_res != 0) {
        free(bwt_out);
        return -2;
    }

    // Этап 2: RLE
    uint8_t *rle_out = (uint8_t *)malloc(input_size * 2); // worst case
    if (!rle_out) {
        free(bwt_out);
        return -10;
    }

    size_t rle_size = 0;
    int rle_res = (mode == MODE_CUDA)
        ? rle_compress_cuda(bwt_out, input_size, rle_out, &rle_size)
        : rle_compress_cpu(bwt_out, input_size, rle_out, &rle_size);

    free(bwt_out);
    if (rle_res != 0) {
        free(rle_out);
        return -3;
    }

    // Этап 3: Huffman
    uint8_t *prepended = (uint8_t *)malloc(rle_size + 4);
    if (!prepended) {
        free(rle_out);
        return -10;
    }

    // Сохраняем индекс BWT (4 байта)
    prepended[0] = (uint8_t)(primary_index & 0xFF);
    prepended[1] = (uint8_t)((primary_index >> 8) & 0xFF);
    prepended[2] = (uint8_t)((primary_index >> 16) & 0xFF);
    prepended[3] = (uint8_t)((primary_index >> 24) & 0xFF);
    memcpy(prepended + 4, rle_out, rle_size);
    free(rle_out);

    int res = huffman_compress_cpu(prepended, rle_size + 4, output, output_size);
    free(prepended);
    return res;
}

int decompress_bwt_rle_huffman(const uint8_t *input, size_t input_size,
                                uint8_t *output, size_t *output_size,
                                ExecutionMode mode) {
    if (!input || !output || !output_size) return -1;

    // Этап 1: Huffman
    uint8_t *huffman_out = (uint8_t *)malloc(MAX_SIZE);
    if (!huffman_out) return -10;

    size_t huffman_size = MAX_SIZE;
    int res1 = huffman_decompress_cpu(input, input_size, huffman_out, &huffman_size);
    if (res1 != 0) {
        free(huffman_out);
        return -2;
    }

    // Извлекаем индекс BWT (4 байта)
    size_t primary_index = 0;
    primary_index |= huffman_out[0];
    primary_index |= ((int)huffman_out[1]) << 8;
    primary_index |= ((int)huffman_out[2]) << 16;
    primary_index |= ((int)huffman_out[3]) << 24;

    // Этап 2: RLE
    uint8_t *rle_input = huffman_out + 4;
    size_t rle_input_size = huffman_size - 4;

    uint8_t *rle_out = (uint8_t *)malloc(MAX_SIZE);
    if (!rle_out) {
        free(huffman_out);
        return -10;
    }

    size_t rle_out_size = MAX_SIZE;
    int rle_res = (mode == MODE_CUDA)
        ? rle_decompress_cuda(rle_input, rle_input_size, rle_out, &rle_out_size)
        : rle_decompress_cpu(rle_input, rle_input_size, rle_out, &rle_out_size);

    free(huffman_out);
    if (rle_res != 0) {
        free(rle_out);
        return -3;
    }

    // Этап 3: BWT
    int res = (mode == MODE_CUDA)
        ? bwt_inverse_cuda(rle_out, rle_out_size, output, primary_index)
        : bwt_inverse_cpu(rle_out, rle_out_size, output, primary_index);

    *output_size = rle_out_size;
    free(rle_out);
    return res;
}
