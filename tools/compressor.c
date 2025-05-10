#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/compression.h"

#define MAX_FILE_SIZE (10 * 1024 * 1024) // 10 MB

void print_help(const char *prog) {
    printf("Compression Utility\n\n");
    printf("Usage:\n");
    printf("  %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --algo <name>        Compression algorithm [rle | lz77 | lzw | huffman | bwt-rle-huffman] (default: rle)\n");
    printf("  --mode <type>        Execution mode [cpu | cuda] (default: cpu)\n");
    printf("  --compress           Perform compression (default if --decompress is not specified)\n");
    printf("  --decompress         Perform decompression (overrides --compress)\n");
    printf("  --input <filename>   Input file (auto: test.txt or output.bin)\n");
    printf("  --output <filename>  Output file (auto: output.bin or restored.txt)\n");
    printf("  --help               Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s --algo rle --compress --input data/a.txt --output a.rle\n", prog);
    printf("  %s --algo rle --decompress --input a.rle --output a.txt\n", prog);
}

int parse_algo(const char *str) {
    if (strcmp(str, "rle") == 0) return ALGO_RLE;
    if (strcmp(str, "lz77") == 0) return ALGO_LZ77;
    if (strcmp(str, "lzw") == 0) return ALGO_LZW;
    if (strcmp(str, "huffman") == 0) return ALGO_HUFFMAN;
    if (strcmp(str, "bwt-rle-huffman") == 0) return ALGO_BWT_RLE_HUFFMAN;
    return -1;
}

int parse_mode(const char *str) {
    if (strcmp(str, "cpu") == 0) return MODE_CPU;
    if (strcmp(str, "cuda") == 0) return MODE_CUDA;
    return -1;
}

int read_file(const char *filename, uint8_t *buffer, size_t *size) {
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    *size = fread(buffer, 1, MAX_FILE_SIZE, f);
    fclose(f);
    return 0;
}

int write_file(const char *filename, const uint8_t *buffer, size_t size) {
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    fwrite(buffer, 1, size, f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    // Значения по умолчанию
    const char *algo_str = "rle";
    const char *mode_str = "cpu";
    const char *input_file = NULL;
    const char *output_file = NULL;
    int compress = 1, decompress = 0;

    // Разбор аргументов
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            algo_str = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (strcmp(argv[i], "--compress") == 0) {
            compress = 1;
        } else if (strcmp(argv[i], "--decompress") == 0) {
            decompress = 1;
            compress = 0;
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    // Подстановка input/output по умолчанию
    if (!input_file) {
        input_file = decompress ? "data/output.bin" : "data/test.txt";
    }

    if (!output_file) {
        output_file = decompress ? "data/restored.txt" : "data/output.bin";
    }

    int algo_val = parse_algo(algo_str);
    int mode_val = parse_mode(mode_str);

    if (algo_val == -1 || mode_val == -1) {
        fprintf(stderr, "Unknown algorithm or mode.\n");
        return 1;
    }

    CompressionAlgo algo = (CompressionAlgo)algo_val;
    ExecutionMode mode = (ExecutionMode)mode_val;

    uint8_t *input = malloc(MAX_FILE_SIZE);
    uint8_t *output = malloc(MAX_FILE_SIZE);
    size_t input_size = 0, output_size = 0;

    if (!input || !output) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    if (read_file(input_file, input, &input_size) != 0) {
        fprintf(stderr, "Failed to read input file: %s\n", input_file);
        return 1;
    }

    int result = 0;
    if (compress)
        result = compress_data(input, input_size, output, &output_size, algo, mode);
    else
        result = decompress_data(input, input_size, output, &output_size, algo, mode);

    if (result != 0) {
        fprintf(stderr, "Compression/decompression failed (code %d).\n", result);
        return 1;
    }

    if (write_file(output_file, output, output_size) != 0) {
        fprintf(stderr, "Failed to write output file: %s\n", output_file);
        return 1;
    }

    printf("Done. Output written to %s (%zu bytes)\n", output_file, output_size);

    free(input);
    free(output);
    return 0;
}
