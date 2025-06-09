#ifndef LZW_H
#define LZW_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void lzw_compress_cpu(const char* input_file, const char* output_file);

void lzw_decompress_cpu(const char* input_file, const char* output_file);

#ifdef __cplusplus
}
#endif

#endif // LZW_H
