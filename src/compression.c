#include "compression.h"
#include "bwt.h"
#include "rle.h"
#include "lz77.h"
#include "lzw.h"
#include "huffman.h"
#include "bwt_rle_huffman.h"

int compress_data(const uint8_t *input, size_t input_size,
                  uint8_t *output, size_t *output_size,
                  CompressionAlgo algo, ExecutionMode mode) {
    if (!input || !output || !output_size) return -1;

    switch (algo) {
        case ALGO_RLE:
            if (mode == MODE_CPU)
                return rle_compress_cpu(input, input_size, output, output_size);
            else if (mode == MODE_CUDA)
                return rle_compress_cuda(input, input_size, output, output_size);
            break;
        
        case ALGO_LZ77:
            if (mode == MODE_CPU)
                return lz77_compress_cpu(input, input_size, output, output_size);
            break;

        case ALGO_LZW:
            if (mode == MODE_CPU)
                return lzw_compress_cpu(input, input_size, output, output_size);
            break;
        
        case ALGO_HUFFMAN:
            if (mode == MODE_CPU)
                return huffman_compress_cpu(input, input_size, output, output_size);
            else if (mode == MODE_CUDA)
                return huffman_compress_cuda(input, input_size, output, output_size);
            break;
        
        case ALGO_BWT_RLE_HUFFMAN:
            return compress_bwt_rle_huffman(input, input_size, output, output_size, mode);

        default:
            return -2; // неподдерживаемый алгоритм
    }
    return -3; // неподдерживаемый режим
}

int decompress_data(const uint8_t *input, size_t input_size,
                    uint8_t *output, size_t *output_size,
                    CompressionAlgo algo, ExecutionMode mode) {
    if (!input || !output || !output_size) return -1;

    switch (algo) {
        case ALGO_RLE:
            if (mode == MODE_CPU)
                return rle_decompress_cpu(input, input_size, output, output_size);
            else if (mode == MODE_CUDA)
                return rle_decompress_cuda(input, input_size, output, output_size);
            break;
        
        case ALGO_LZ77:
            if (mode == MODE_CPU)
                return lz77_decompress_cpu(input, input_size, output, output_size);
            break;
    
        case ALGO_LZW:
            if (mode == MODE_CPU)
                return lzw_decompress_cpu(input, input_size, output, output_size);
            break;

        case ALGO_HUFFMAN:
            if (mode == MODE_CPU)
                return huffman_decompress_cpu(input, input_size, output, output_size);
            else if (mode == MODE_CUDA)
                return huffman_decompress_cpu(input, input_size, output, output_size);
            break;
        
        case ALGO_BWT_RLE_HUFFMAN:
            return decompress_bwt_rle_huffman(input, input_size, output, output_size, mode);

        default:
            return -2;
    }
    return -3;
}
