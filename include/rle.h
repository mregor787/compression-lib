#ifndef RLE_H
#define RLE_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU
void rle_compress_cpu(const char* input_file, const char* output_file);

void rle_decompress_cpu(const char* input_file, const char* output_file);

// CUDA
void rle_compress_cuda(const char* input_file, const char* output_file);

void rle_decompress_cuda(const char* input_file, const char* output_file);

#ifdef __cplusplus
}
#endif

#endif // RLE_H
