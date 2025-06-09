#ifndef LZ77_H
#define LZ77_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

void lz77_compress_cpu(const char* input_file, const char* output_file);

void lz77_decompress_cpu(const char* input_file, const char* output_file);

#ifdef __cplusplus
}
#endif

#endif // LZ77_H
