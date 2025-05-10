#ifndef LZ77_H
#define LZ77_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int lz77_compress_cpu(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *output_size);

int lz77_decompress_cpu(const uint8_t *input, size_t input_size,
                        uint8_t *output, size_t *output_size);

#ifdef __cplusplus
}
#endif

#endif // LZ77_H
