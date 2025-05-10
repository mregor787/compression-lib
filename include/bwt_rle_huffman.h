#ifndef BWT_RLE_HUFFMAN_H
#define BWT_RLE_HUFFMAN_H

#include <stdint.h>
#include <stddef.h>
#include "compression.h"

#ifdef __cplusplus
extern "C" {
#endif

int compress_bwt_rle_huffman(const uint8_t *input, size_t input_size,
                             uint8_t *output, size_t *output_size,
                             ExecutionMode mode);

int decompress_bwt_rle_huffman(const uint8_t *input, size_t input_size,
                               uint8_t *output, size_t *output_size,
                               ExecutionMode mode);

#ifdef __cplusplus
}
#endif

#endif // BWT_RLE_HUFFMAN_H
