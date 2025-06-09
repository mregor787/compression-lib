#ifndef BWT_RLE_HUFFMAN_H
#define BWT_RLE_HUFFMAN_H

#include <stdint.h>
#include <stddef.h>
#include "compression.h"

#ifdef __cplusplus
extern "C" {
#endif

void compress_bwt_rle_huffman(const char* input_file, const char* output_file, ExecutionMode mode);

void decompress_bwt_rle_huffman(const char* input_file, const char* output_file, ExecutionMode mode);

#ifdef __cplusplus
}
#endif

#endif // BWT_RLE_HUFFMAN_H
