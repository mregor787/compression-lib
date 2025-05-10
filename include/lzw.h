#ifndef LZW_H
#define LZW_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int lzw_compress_cpu(const uint8_t *input, size_t input_size,
                     uint8_t *output, size_t *output_size);

int lzw_decompress_cpu(const uint8_t *input, size_t input_size,
                       uint8_t *output, size_t *output_size);

#ifdef __cplusplus
}
#endif

#endif // LZW_H
