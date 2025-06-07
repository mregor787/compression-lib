#include "compression.h"
#include "bwt.h"
#include "rle.h"
#include "lz77.h"
#include "lzw.h"
#include "huffman.h"
#include "bwt_rle_huffman.h"

int compress_data(const char* input_file, const char* output_file,
                  CompressionAlgo algo, ExecutionMode mode) 
{
    if (!input_file || !output_file) return -1;

    switch (algo) {
        case ALGO_RLE:
            if (mode == MODE_CPU)
                rle_compress_cpu(input_file, output_file);
            else if (mode == MODE_CUDA)
                rle_compress_cuda(input_file, output_file);
            break;
/*        
        case ALGO_LZ77:
            if (mode == MODE_CPU)
                lz77_compress_cpu(input_file, output_file);
            break;

        case ALGO_LZW:
            if (mode == MODE_CPU)
                lzw_compress_cpu(input_file, output_file);
            break;
        
        case ALGO_HUFFMAN:
            if (mode == MODE_CPU)
                huffman_compress_cpu(input_file, output_file);
            else if (mode == MODE_CUDA)
                huffman_compress_cuda(input_file, output_file);
            break;
        
        case ALGO_BWT_RLE_HUFFMAN:
            compress_bwt_rle_huffman(input_file, output_file, mode);
*/
        default:
            return -2; // неподдерживаемый алгоритм
    }
    return -3; // неподдерживаемый режим
}

int decompress_data(const char* input_file, const char* output_file,
                    CompressionAlgo algo, ExecutionMode mode) 
{
    if (!input_file || !output_file) return -1;

    switch (algo) {
        case ALGO_RLE:
            if (mode == MODE_CPU)
                rle_decompress_cpu(input_file, output_file);
            else if (mode == MODE_CUDA)
                rle_decompress_cuda(input_file, output_file);
            break;
/*
        case ALGO_LZ77:
            if (mode == MODE_CPU)
                lz77_decompress_cpu(input_file, output_file);
            break;
    
        case ALGO_LZW:
            if (mode == MODE_CPU)
                lzw_decompress_cpu(input_file, output_file);
            break;

        case ALGO_HUFFMAN:
            if (mode == MODE_CPU)
                huffman_decompress_cpu(input_file, output_file);
            else if (mode == MODE_CUDA)
                huffman_decompress_cpu(input_file, output_file);
            break;
        
        case ALGO_BWT_RLE_HUFFMAN:
            decompress_bwt_rle_huffman(input_file, output_file, mode);
*/
        default:
            return -2;
    }
    return -3;
}
