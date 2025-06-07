#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <ctype.h>

#include "../include/compression.h"

void print_help(const char *prog) {
    printf("Compression Utility\n\n");
    printf("Usage:\n");
    printf("  %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --algo <name>        Compression algorithm [rle | lz77 | lzw | huffman | bwt-rle-huffman] (default: rle)\n");
    printf("  --mode <type>        Execution mode [cpu | cuda] (default: cpu)\n");
    printf("  --compress           Perform compression (default if --decompress is not specified)\n");
    printf("  --decompress         Perform decompression (overrides --compress)\n");
    printf("  --input <filename>   Input file (default: test.txt or output.bin)\n");
    printf("  --output <filename>  Output file (default: output.bin or restored.txt)\n");
    printf("  --benchmark          Measure time for compress + decompress, ignore other mode flags\n");
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

size_t get_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "Cannot stat file: %s\n", path);
        return 0;
    }
    return (size_t)st.st_size;
}

int files_are_equal(const char* path1, const char* path2) {
    FILE* f1 = fopen(path1, "rb");
    FILE* f2 = fopen(path2, "rb");
    if (!f1 || !f2) {
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        return 0;
    }

    fseek(f1, 0, SEEK_END);
    fseek(f2, 0, SEEK_END);
    size_t size1 = ftell(f1);
    size_t size2 = ftell(f2);
    if (size1 != size2) {
        fclose(f1);
        fclose(f2);
        return 0;
    }
    rewind(f1);
    rewind(f2);

    int equal = 1;
    for (size_t i = 0; i < size1; ++i) {
        int c1 = fgetc(f1);
        int c2 = fgetc(f2);
        if (c1 != c2) {
            equal = 0;
            break;
        }
    }

    fclose(f1);
    fclose(f2);
    return equal;
}

void print_report_header(const char* algo_str)
{
    char upper[64];
    size_t i = 0;

    for (; algo_str[i] && i < sizeof(upper) - 1; ++i)
        upper[i] = toupper((unsigned char)algo_str[i]);

    upper[i] = '\0';

    printf("\n=== %s Compression Report ===\n", upper);
}

int main(int argc, char **argv) {
    const char *algo_str = "rle";
    const char *mode_str = "cpu";
    const char *input_file = NULL;
    const char *output_file = NULL;
    int compress = 1, decompress = 0, benchmark = 0;

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
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark = 1;
        }
    }

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

    if (benchmark) {
        const char* input = "data/test.txt";
        const char* output = "data/output.bin";
        const char* restored = "data/restored.txt";

        compress_data(input, output, algo, mode);
        decompress_data(output, restored, algo, mode);

        size_t original_size = get_file_size(input);
        size_t compressed_size = get_file_size(output);
        size_t restored_size = get_file_size(restored);

        int match = files_are_equal(input, restored);

        print_report_header(algo_str);
        printf("Original file:    %s (%zu bytes)\n", input, original_size);
        printf("Compressed file:  %s (%zu bytes)\n", output, compressed_size);
        printf("Restored file:    %s (%zu bytes)\n", restored, restored_size);
        printf("Compression ratio: %.2fx\n", compressed_size ? (double)original_size / compressed_size : 0.0);
        printf("Restored file matches original: %s\n", match ? "YES" : "NO");
        return 0;
    }

    if (compress)
        compress_data(input_file, output_file, algo, mode);
    else
        decompress_data(input_file, output_file, algo, mode);

    return 0;
}
