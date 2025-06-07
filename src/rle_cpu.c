#include "rle.h"

void rle_compress_cpu(const char* input_file, const char* output_file)
{
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Cannot open input file: %s\n", input_file);
        return;
    }

    fseek(in, 0, SEEK_END);
    size_t filesize = ftell(in);
    fseek(in, 0, SEEK_SET);

    uint8_t* input = (uint8_t*)malloc(filesize);
    fread(input, 1, filesize, in);
    fclose(in);

    uint8_t* output_data = (uint8_t*)malloc(filesize);
    uint32_t* occurences = (uint32_t*)malloc(filesize * sizeof(uint32_t));
    uint32_t group_count = 0;

    clock_t start = clock();

    size_t i = 0;
    while (i < filesize) {
        uint8_t current = input[i];
        size_t count = 1;
        while (i + count < filesize && input[i + count] == current) {
            ++count;
        }
        output_data[group_count] = current;
        occurences[group_count] = (uint32_t)count;
        group_count++;
        i += count;
    }

    clock_t end = clock();
    printf("%s CPU compression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        free(input); free(output_data); free(occurences);
        return;
    }

    fwrite(&group_count, sizeof(uint32_t), 1, out);
    fwrite(output_data, sizeof(uint8_t), group_count, out);
    fwrite(occurences, sizeof(uint32_t), group_count, out);
    fclose(out);

    free(input);
    free(output_data);
    free(occurences);
}

void rle_decompress_cpu(const char* input_file, const char* output_file)
{
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Cannot open input file: %s\n", input_file);
        return;
    }

    uint32_t group_count;
    fread(&group_count, sizeof(uint32_t), 1, in);

    uint8_t* input_data = (uint8_t*)malloc(group_count);
    uint32_t* occurences = (uint32_t*)malloc(group_count * sizeof(uint32_t));
    fread(input_data, sizeof(uint8_t), group_count, in);
    fread(occurences, sizeof(uint32_t), group_count, in);
    fclose(in);

    size_t output_size = 0;
    for (uint32_t i = 0; i < group_count; ++i)
        output_size += occurences[i];

    uint8_t* output_data = (uint8_t*)malloc(output_size);

    clock_t start = clock();

    size_t pos = 0;
    for (uint32_t i = 0; i < group_count; ++i) {
        memset(output_data + pos, input_data[i], occurences[i]);
        pos += occurences[i];
    }

    clock_t end = clock();
    printf("%s CPU decompression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        free(input_data); free(occurences); free(output_data);
        return;
    }

    fwrite(output_data, 1, output_size, out);
    fclose(out);

    free(input_data);
    free(occurences);
    free(output_data);
}
