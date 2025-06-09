#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    ALGO_RLE,
    ALGO_LZ77,
    ALGO_LZW,
    ALGO_BWT,
    ALGO_HUFFMAN,
    ALGO_BWT_RLE_HUFFMAN
} CompressionAlgo;

typedef enum {
    MODE_CPU,
    MODE_CUDA
} ExecutionMode;

int compress_data(const char* input_file, const char* output_file,
                  CompressionAlgo algo, ExecutionMode mode);

int decompress_data(const char* input_file, const char* output_file,
                    CompressionAlgo algo, ExecutionMode mode);

#endif // COMPRESSION_H
