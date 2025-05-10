#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU
int huffman_compress_cpu(const uint8_t *input, size_t input_size,
                         uint8_t *output, size_t *output_size);

int huffman_decompress_cpu(const uint8_t *input, size_t input_size,
                           uint8_t *output, size_t *output_size);

// CUDA
int huffman_compress_cuda(const uint8_t *input, size_t input_size,
                          uint8_t *output, size_t *output_size);

// Только для внутреннего использования (подсчёт частот на GPU)
int count_frequencies_cuda(const uint8_t *input, size_t input_size, int *freq_out);

#ifdef __cplusplus
}
#endif

#endif // HUFFMAN_H
