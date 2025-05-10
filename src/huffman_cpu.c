#include "huffman.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#define MAX_NODES 512

typedef struct HuffmanNode {
    uint8_t value;
    uint32_t freq;
    struct HuffmanNode *left;
    struct HuffmanNode *right;
} HuffmanNode;

typedef struct {
    HuffmanNode *nodes[MAX_NODES];
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
            HuffmanNode *node = malloc(sizeof(HuffmanNode));
            node->value = (uint8_t)i;
            node->freq = freq[i];
            node->left = node->right = NULL;
            heap_push(&heap, node);
        }
    }

    while (heap.size > 1) {
        HuffmanNode *a = heap_pop(&heap);
        HuffmanNode *b = heap_pop(&heap);

        HuffmanNode *parent = malloc(sizeof(HuffmanNode));
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

int huffman_compress_cpu(const uint8_t *input, size_t input_size,
                         uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size) return -1;

    uint32_t freq[256] = {0};
    for (size_t i = 0; i < input_size; ++i)
        freq[input[i]]++;

    HuffmanNode *root = build_huffman_tree(freq);
    if (!root) return -2;

    HuffmanCode table[256] = {0};
    build_code_table(root, table, 0, 0);

    // 1. Сохраняем таблицу частот (1024 байта)
    memcpy(output, freq, 256 * sizeof(uint32_t));
    size_t byte_pos = 256 * sizeof(uint32_t);
    uint8_t bit_buffer = 0;
    int bit_count = 0;

    // 2. Кодируем поток
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

    // если остались биты
    if (bit_count > 0) {
        bit_buffer <<= (8 - bit_count);
        output[byte_pos++] = bit_buffer;
    }

    // 3. Добавим размер исходных данных в конец (4 байта)
    output[byte_pos++] = (uint8_t)(input_size & 0xFF);
    output[byte_pos++] = (uint8_t)((input_size >> 8) & 0xFF);
    output[byte_pos++] = (uint8_t)((input_size >> 16) & 0xFF);
    output[byte_pos++] = (uint8_t)((input_size >> 24) & 0xFF);

    *output_size = byte_pos;
    free_tree(root);
    return 0;
}

int huffman_decompress_cpu(const uint8_t *input, size_t input_size,
                           uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size < 1028)
        return -1;

    // 1. Восстановим таблицу частот
    uint32_t freq[256];
    memcpy(freq, input, 256 * sizeof(uint32_t));

    // 2. Восстановим размер оригинала
    uint32_t original_size = 0;
    size_t length_pos = input_size - 4;
    original_size |= input[length_pos];
    original_size |= ((uint32_t)input[length_pos + 1]) << 8;
    original_size |= ((uint32_t)input[length_pos + 2]) << 16;
    original_size |= ((uint32_t)input[length_pos + 3]) << 24;

    HuffmanNode *root = build_huffman_tree(freq);
    if (!root) return -2;

    size_t in_pos = 256 * sizeof(uint32_t);
    HuffmanNode *node = root;
    size_t out_pos = 0;

    while (in_pos < length_pos) {
        uint8_t byte = input[in_pos++];
        for (int bit_index = 7; bit_index >= 0; --bit_index) {
            int bit = (byte >> bit_index) & 1;
            node = bit == 0 ? node->left : node->right;

            if (!node->left && !node->right) {
                output[out_pos++] = node->value;
                node = root;
                if (out_pos == original_size) break;
            }
        }
        if (out_pos == original_size) break;
    }

    free_tree(root);
    *output_size = out_pos;
    return 0;
}
