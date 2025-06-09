#ifndef BWT_H
#define BWT_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU
void build_suffix_array(const char *s, int n, int *sa);

void bwt_transform_cpu(const char *input_file, const char *output_file);

void bwt_inverse_cpu(const char *input_file, const char *output_file);
 
// CUDA
void bwt_transform_cuda(const char *input_file, const char *output_file);

void bwt_inverse_cuda(const char *input_file, const char *output_file);


#ifdef __cplusplus
}
#endif

#endif // BWT_H
