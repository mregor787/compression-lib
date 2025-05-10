#ifndef BWT_H
#define BWT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU
int bwt_transform_cpu(const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *primary_index);

int bwt_inverse_cpu(const uint8_t *input, size_t input_size,
                    uint8_t *output, size_t primary_index);
 
// CUDA
int bwt_transform_cuda(const uint8_t *input, size_t n,
                       uint8_t *output, size_t *primary_index);

int bwt_inverse_cuda(const uint8_t *input, size_t n,
                     uint8_t *output, size_t primary_index);


#ifdef __cplusplus
}
#endif

#endif // BWT_H
