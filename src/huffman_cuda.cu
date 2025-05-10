#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "huffman.h"

// ---------- GPU-часть ----------

__global__ void count_kernel(const uint8_t *input, size_t size, int *freq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&freq[input[i]], 1);
    }
}

int count_frequencies_cuda(const uint8_t *input, size_t input_size, int *freq_out) {
    if (!input || !freq_out || input_size == 0)
        return -1;

    uint8_t *d_input;
    int *d_freq;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_freq, 256 * sizeof(int));
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_freq, 0, 256 * sizeof(int));

    int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    count_kernel<<<blocks, threads>>>(d_input, input_size, d_freq);
    cudaDeviceSynchronize();

    cudaMemcpy(freq_out, d_freq, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_freq);

    return 0;
}

// ---------- CPU-часть (reuse из huffman_cpu.c) ----------

typedef struct HuffmanNode {
    uint8_t value;
    uint32_t freq;
    struct HuffmanNode *left;
    struct HuffmanNode *right;
} HuffmanNode;

typedef struct {
    HuffmanNode *nodes[512];
    int size;
} MinHeap;

static void heap_push(MinHeap *heap, HuffmanNode *node) {
    int i = heap->size++;
    while (i > 0 && heap->nodes[(i - 1) / 2]->freq > node->freq) {
        heap->nodes[i] = heap->nodes[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    heap->nodes[i] = node;
}

static HuffmanNode *heap_pop(MinHeap *heap) {
    HuffmanNode *res = heap->nodes[0];
    HuffmanNode *last = heap->nodes[--heap->size];
    int i = 0;
    while (i * 2 + 1 < heap->size) {
        int child = i * 2 + 1;
        if (child + 1 < heap->size &&
            heap->nodes[child + 1]->freq < heap->nodes[child]->freq)
            child++;
        if (last->freq <= heap->nodes[child]->freq)
            break;
        heap->nodes[i] = heap->nodes[child];
        i = child;
    }
    heap->nodes[i] = last;
    return res;
}

static HuffmanNode *build_huffman_tree(uint32_t freq[256]) {
    MinHeap heap = { .size = 0 };
    for (int i = 0; i < 256; ++i) {
        if (freq[i]) {
            HuffmanNode *node = (HuffmanNode *)malloc(sizeof(HuffmanNode));
            node->value = (uint8_t)i;
            node->freq = freq[i];
            node->left = node->right = NULL;
            heap_push(&heap, node);
        }
    }

    while (heap.size > 1) {
        HuffmanNode *a = heap_pop(&heap);
        HuffmanNode *b = heap_pop(&heap);
        HuffmanNode *parent = (HuffmanNode *)malloc(sizeof(HuffmanNode));
        parent->value = 0;
        parent->freq = a->freq + b->freq;
        parent->left = a;
        parent->right = b;
        heap_push(&heap, parent);
    }

    return heap.size > 0 ? heap.nodes[0] : NULL;
}

typedef struct {
    uint32_t bits;
    uint8_t length;
} HuffmanCode;

static void build_code_table(HuffmanNode *node,
                             HuffmanCode table[256],
                             uint32_t code, uint8_t length) {
    if (!node->left && !node->right) {
        table[node->value].bits = code;
        table[node->value].length = length;
        return;
    }

    if (node->left)
        build_code_table(node->left, table, (code << 1), length + 1);
    if (node->right)
        build_code_table(node->right, table, (code << 1) | 1, length + 1);
}

static void free_tree(HuffmanNode *node) {
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

int huffman_compress_cuda(const uint8_t *input, size_t input_size,
                          uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size == 0)
        return -1;

    // 1. Подсчёт частот на GPU
    int freq_temp[256] = {0};
    if (count_frequencies_cuda(input, input_size, freq_temp) != 0)
        return -2;

    uint32_t freq[256];
    for (int i = 0; i < 256; ++i)
        freq[i] = (uint32_t)freq_temp[i];

    // 2. Построение дерева и таблицы кодов
    HuffmanNode *root = build_huffman_tree(freq);
    if (!root) return -3;

    HuffmanCode table[256] = {0};
    build_code_table(root, table, 0, 0);

    // 3. Сохраняем таблицу частот
    memcpy(output, freq, 256 * sizeof(uint32_t));
    size_t byte_pos = 256 * sizeof(uint32_t);
    uint8_t bit_buffer = 0;
    int bit_count = 0;

    // 4. Запись битов
    for (size_t i = 0; i < input_size; ++i) {
        HuffmanCode code = table[input[i]];
        for (int j = code.length - 1; j >= 0; --j) {
            bit_buffer <<= 1;
            bit_buffer |= (code.bits >> j) & 1;
            bit_count++;

            if (bit_count == 8) {
                output[byte_pos++] = bit_buffer;
                bit_buffer = 0;
                bit_count = 0;
            }
        }
    }

    if (bit_count > 0) {
        bit_buffer <<= (8 - bit_count);
        output[byte_pos++] = bit_buffer;
    }

    // 5. Добавим размер исходных данных
    output[byte_pos++] = (uint8_t)(input_size & 0xFF);
    output[byte_pos++] = (uint8_t)((input_size >> 8) & 0xFF);
    output[byte_pos++] = (uint8_t)((input_size >> 16) & 0xFF);
    output[byte_pos++] = (uint8_t)((input_size >> 24) & 0xFF);

    *output_size = byte_pos;
    free_tree(root);
    return 0;
}
