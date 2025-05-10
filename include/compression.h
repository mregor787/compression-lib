#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    ALGO_RLE,
    ALGO_LZ77,
    ALGO_LZW,
    ALGO_HUFFMAN,
    ALGO_BWT_RLE_HUFFMAN
} CompressionAlgo;

typedef enum {
    MODE_CPU,
    MODE_CUDA
} ExecutionMode;

int compress_data(const uint8_t *input, size_t input_size,
                  uint8_t *output, size_t *output_size,
                  CompressionAlgo algo, ExecutionMode mode);

int decompress_data(const uint8_t *input, size_t input_size,
                    uint8_t *output, size_t *output_size,
                    CompressionAlgo algo, ExecutionMode mode);

#endif // COMPRESSION_H
