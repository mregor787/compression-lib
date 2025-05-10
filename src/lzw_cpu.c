#include "lzw.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_DICT_SIZE 4096

// ==== Общая структура словаря ====

typedef struct {
    char **entries;
    int size;
    int capacity;
} Dict;

static Dict *dict_create(int capacity) {
    Dict *d = (Dict *)malloc(sizeof(Dict));
    d->entries = (char **)malloc(sizeof(char *) * capacity);
    d->size = 256;
    d->capacity = capacity;
    for (int i = 0; i < 256; ++i) {
        d->entries[i] = (char *)malloc(2);
        d->entries[i][0] = (char)i;
        d->entries[i][1] = '\0';
    }
    return d;
}

static void dict_add(Dict *d, const char *str) {
    if (d->size >= d->capacity) return;
    d->entries[d->size++] = strdup(str);
}

static int dict_find(Dict *d, const char *str) {
    for (int i = 0; i < d->size; ++i) {
        if (strcmp(d->entries[i], str) == 0)
            return i;
    }
    return -1;
}

static void dict_free(Dict *d) {
    for (int i = 0; i < d->size; ++i)
        free(d->entries[i]);
    free(d->entries);
    free(d);
}

// ==== Сжатие ====

int lzw_compress_cpu(const uint8_t *input, size_t input_size,
                     uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size) return -1;

    Dict *dict = dict_create(MAX_DICT_SIZE);
    char buffer[512] = {0};

    size_t in_pos = 0, out_pos = 0;
    size_t buf_len = 1;
    buffer[0] = input[in_pos++];

    while (in_pos <= input_size) {
        buffer[buf_len] = input[in_pos];
        buffer[buf_len + 1] = '\0';
        int idx = dict_find(dict, buffer);
        if (idx != -1 && in_pos < input_size) {
            buf_len++;
            in_pos++;
        } else {
            buffer[buf_len] = '\0';
            int code = dict_find(dict, buffer);
            output[out_pos++] = code & 0xFF;
            output[out_pos++] = (code >> 8) & 0xFF;

            if (in_pos < input_size) {
                buffer[buf_len] = input[in_pos];
                buffer[buf_len + 1] = '\0';
                dict_add(dict, buffer);
                buf_len = 1;
                buffer[0] = input[in_pos++];
            } else {
                break;
            }
        }
    }

    *output_size = out_pos;
    dict_free(dict);
    return 0;
}

// ==== Распаковка ====

int lzw_decompress_cpu(const uint8_t *input, size_t input_size,
                       uint8_t *output, size_t *output_size) {
    if (!input || !output || !output_size || input_size % 2 != 0) return -1;

    Dict *dict = dict_create(MAX_DICT_SIZE);
    char temp[1024] = {0};

    size_t in_pos = 0, out_pos = 0;
    uint16_t prev_code = input[in_pos] | (input[in_pos + 1] << 8);
    strcpy(temp, dict->entries[prev_code]);
    size_t len = strlen(temp);
    memcpy(output + out_pos, temp, len);
    out_pos += len;
    in_pos += 2;

    while (in_pos + 1 < input_size) {
        uint16_t curr_code = input[in_pos] | (input[in_pos + 1] << 8);
        in_pos += 2;

        const char *entry = NULL;
        if (curr_code < dict->size) {
            entry = dict->entries[curr_code];
        } else if (curr_code == dict->size) {
            snprintf(temp, sizeof(temp), "%s%c", dict->entries[prev_code], dict->entries[prev_code][0]);
            entry = temp;
        } else {
            dict_free(dict);
            return -2;
        }

        size_t entry_len = strlen(entry);
        memcpy(output + out_pos, entry, entry_len);
        out_pos += entry_len;

        snprintf(temp, sizeof(temp), "%s%c", dict->entries[prev_code], entry[0]);
        dict_add(dict, temp);
        prev_code = curr_code;
    }

    *output_size = out_pos;
    dict_free(dict);
    return 0;
}
