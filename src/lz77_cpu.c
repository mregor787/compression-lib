#include <lz77.h>

#define WINDOW_SIZE 4096
#define LOOKAHEAD_SIZE 18

void lz77_compress_cpu(const char* input_file, const char* output_file)
{
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Cannot open input file: %s\n", input_file);
        return;
    }

    fseek(in, 0, SEEK_END);
    size_t input_size = ftell(in);
    fseek(in, 0, SEEK_SET);

    uint8_t* input = (uint8_t*)malloc(input_size);
    fread(input, 1, input_size, in);
    fclose(in);

    uint8_t* output = (uint8_t*)malloc(input_size * 4);
    size_t in_pos = 0, out_pos = 0;

    clock_t start = clock();

    while (in_pos < input_size) {
        size_t best_offset = 0;
        size_t best_length = 0;

        size_t window_start = (in_pos >= WINDOW_SIZE) ? (in_pos - WINDOW_SIZE) : 0;

        for (size_t j = window_start; j < in_pos; ++j) {
            size_t length = 0;
            while (length < LOOKAHEAD_SIZE &&
                   j + length < in_pos &&
                   in_pos + length < input_size &&
                   input[j + length] == input[in_pos + length]) {
                ++length;
            }

            if (length > best_length) {
                best_length = length;
                best_offset = in_pos - j;
            }
        }

        if (in_pos + best_length >= input_size) {
            best_length = 0;
            best_offset = 0;
        }

        uint8_t next_symbol = input[in_pos + best_length];

        output[out_pos++] = (best_offset >> 8) & 0xFF;
        output[out_pos++] = best_offset & 0xFF;
        output[out_pos++] = (uint8_t)best_length;
        output[out_pos++] = next_symbol;

        in_pos += best_length + 1;
    }

    clock_t end = clock();
    printf("%s LZ77 CPU compression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        free(input);
        free(output);
        return;
    }

    fwrite(output, 1, out_pos, out);
    fclose(out);

    free(input);
    free(output);
}

void lz77_decompress_cpu(const char* input_file, const char* output_file)
{
    FILE* in = fopen(input_file, "rb");
    if (!in) {
        fprintf(stderr, "Cannot open input file: %s\n", input_file);
        return;
    }

    fseek(in, 0, SEEK_END);
    size_t input_size = ftell(in);
    fseek(in, 0, SEEK_SET);

    uint8_t* input = (uint8_t*)malloc(input_size);
    fread(input, 1, input_size, in);
    fclose(in);

    size_t capacity = input_size * 10;
    uint8_t* output = (uint8_t*)malloc(capacity);
    size_t in_pos = 0, out_pos = 0;

    clock_t start = clock();

    while (in_pos + 4 <= input_size) {
        uint16_t offset = ((uint16_t)input[in_pos] << 8) | input[in_pos + 1];
        uint8_t length = input[in_pos + 2];
        uint8_t symbol = input[in_pos + 3];
        in_pos += 4;

        if (offset > 0) {
            if (out_pos < offset) {
                fprintf(stderr, "Decompression error: invalid offset\n");
                free(input);
                free(output);
                return;
            }
            for (size_t i = 0; i < length; ++i) {
                output[out_pos] = output[out_pos - offset];
                ++out_pos;
            }
        }

        output[out_pos++] = symbol;
    }

    clock_t end = clock();
    printf("%s LZ77 CPU decompression time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", output_file);
        free(input);
        free(output);
        return;
    }

    fwrite(output, 1, out_pos, out);
    fclose(out);

    free(input);
    free(output);
}
