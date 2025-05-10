#ifndef RLE_H
#define RLE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU
int rle_compress_cpu(const uint8_t *input, size_t input_size,
                     uint8_t *output, size_t *output_size);

int rle_decompress_cpu(const uint8_t *input, size_t input_size,
                       uint8_t *output, size_t *output_size);

// CUDA
int rle_compress_cuda(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *output_size);

int rle_decompress_cuda(const uint8_t *input, size_t input_size,
                        uint8_t *output, size_t *output_size);

#ifdef __cplusplus
}
#endif

#endif // RLE_H
