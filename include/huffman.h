#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BYTE_RANGE 256

typedef struct Node {
    int byte;
    uint32_t freq;
    struct Node *left, *right;
} Node;

typedef struct {
    uint64_t bits;
    int length;
} HuffCode;

typedef struct {
    Node **data;
    int size;
} MinHeap;

// CPU
void huffman_compress_cpu(const char* input_file, const char* output_file);

void huffman_decompress_cpu(const char* input_file, const char* output_file);

// CUDA
void huffman_compress_cuda(const char* input_file, const char* output_file);


#ifdef __cplusplus
}
#endif

#endif // HUFFMAN_H
