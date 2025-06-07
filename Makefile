# Компиляторы
CC     = gcc
NVCC   = nvcc

# Флаги компиляции
CFLAGS  = -Wall -Wextra -Iinclude
NVFLAGS = -Iinclude --extended-lambda --diag-suppress=20011 -Wno-deprecated-gpu-targets -std=c++14 

# Исходники
SRC       = src/rle_cpu.c src/lz77_cpu.c src/lzw_cpu.c src/huffman_cpu.c src/bwt_cpu.c src/bwt_rle_huffman.c src/compression.c
CUDA_SRC  = src/rle_cuda.cu src/bwt_cuda.cu src/huffman_cuda.cu

# Объекты
OBJ       = $(SRC:.c=.o)
CUDA_OBJ  = $(CUDA_SRC:.cu=.o)

# Целевые файлы
LIB       = libcompression.a
BIN       = compressor

# Основная цель
all: $(LIB) $(BIN)

# Библиотека
$(LIB): $(OBJ) $(CUDA_OBJ)
	ar rcs $@ $^

# Утилита компрессора
compressor: tools/compressor.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^ -lrt

# Автоматическая компиляция .c и .cu файлов в .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Очистка
clean:
	rm -f $(OBJ) $(CUDA_OBJ) $(BIN) $(LIB)
