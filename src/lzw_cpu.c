#include <lzw.h>

#define MAX_DICT_SIZE 4096

typedef struct {
    char **entries;
    int size;
} Dict;

static Dict *dict_create() {
    Dict *dict = (Dict *)malloc(sizeof(Dict));
    dict->entries = (char **)malloc(sizeof(char *) * MAX_DICT_SIZE);
    dict->size = 256;
    for (int i = 0; i < 256; ++i) {
        dict->entries[i] = (char *)malloc(2);
        dict->entries[i][0] = (char)i;
        dict->entries[i][1] = '\0';
    }
    return dict;
}

static void dict_free(Dict *dict) {
    for (int i = 0; i < dict->size; ++i) {
        free(dict->entries[i]);
    }
    free(dict->entries);
    free(dict);
}

static int dict_find(Dict *dict, const char *str) {
    for (int i = 0; i < dict->size; ++i) {
        if (strcmp(dict->entries[i], str) == 0)
            return i;
    }
    return -1;
}

static void dict_add(Dict *dict, const char *str) {
    if (dict->size >= MAX_DICT_SIZE) return;
    dict->entries[dict->size++] = strdup(str);
}

void lzw_compress_cpu(const char *input_file, const char *output_file) {
    FILE *in = fopen(input_file, "rb");
    if (!in) {
        perror("fopen input");
        return;
    }

    fseek(in, 0, SEEK_END);
    long input_size = ftell(in);
    fseek(in, 0, SEEK_SET);

    char *input_data = (char *)malloc(input_size);
    fread(input_data, 1, input_size, in);
    fclose(in);

    clock_t start = clock();

    Dict *dict = dict_create();
    char current[1024] = "";
    FILE *out = fopen(output_file, "wb");

    for (long i = 0; i < input_size; ++i) {
        char c = input_data[i];
        char next[1024];
        snprintf(next, sizeof(next), "%s%c", current, c);

        if (dict_find(dict, next) != -1) {
            strcpy(current, next);
        } else {
            int index = dict_find(dict, current);
            fwrite(&index, sizeof(short), 1, out);
            dict_add(dict, next);
            current[0] = c;
            current[1] = '\0';
        }
    }

    if (strlen(current) > 0) {
        int index = dict_find(dict, current);
        fwrite(&index, sizeof(short), 1, out);
    }

    fclose(out);
    dict_free(dict);
    clock_t end = clock();
    printf("%s LZW CPU compression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    free(input_data);
}

void lzw_decompress_cpu(const char *input_file, const char *output_file) {
    FILE *in = fopen(input_file, "rb");
    if (!in) {
        perror("fopen input");
        return;
    }

    fseek(in, 0, SEEK_END);
    long input_size = ftell(in);
    fseek(in, 0, SEEK_SET);
    int code_count = input_size / sizeof(short);

    short *codes = (short *)malloc(input_size);
    fread(codes, sizeof(short), code_count, in);
    fclose(in);

    clock_t start = clock();

    Dict *dict = dict_create();
    FILE *out = fopen(output_file, "wb");

    char prev[1024] = "";
    for (int i = 0; i < code_count; ++i) {
        short code = codes[i];
        if (code < dict->size) {
            char *entry = dict->entries[code];
            fwrite(entry, 1, strlen(entry), out);

            if (i > 0) {
                char new_entry[1024];
                snprintf(new_entry, sizeof(new_entry), "%s%c", prev, entry[0]);
                dict_add(dict, new_entry);
            }
            strcpy(prev, entry);
        } else {
            char new_entry[1024];
            snprintf(new_entry, sizeof(new_entry), "%s%c", prev, prev[0]);
            fwrite(new_entry, 1, strlen(new_entry), out);
            dict_add(dict, new_entry);
            strcpy(prev, new_entry);
        }
    }

    fclose(out);
    dict_free(dict);
    clock_t end = clock();
    printf("%s LZW CPU decompression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    free(codes);
}
