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
BIN       = compressor test_rle test_lz77 test_lzw test_huffman test_bwt test_bwt_rle_huffman test_rle_cuda test_bwt_cuda test_huffman_cuda

# Основная цель
all: $(LIB) $(BIN)

# Библиотека
$(LIB): $(OBJ) $(CUDA_OBJ)
	ar rcs $@ $^

# Утилита компрессора
compressor: tools/compressor.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^ -lrt

# Тесты
test_rle: tests/test_rle.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^

test_lz77: tests/test_lz77.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $^

test_lzw: tests/test_lzw.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $^

test_huffman: tests/test_huffman.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $^

test_bwt: tests/test_bwt.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $^

test_bwt_rle_huffman: tests/test_bwt_rle_huffman.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^

test_rle_cuda: tests/test_rle_cuda.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^

test_bwt_cuda: tests/test_bwt_cuda.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^

test_huffman_cuda: tests/test_huffman_cuda.c $(LIB)
	$(NVCC) $(NVFLAGS) -o $@ $^

# Автоматическая компиляция .c и .cu файлов в .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Очистка
clean:
	rm -f $(OBJ) $(CUDA_OBJ) $(BIN) $(LIB)

# Тесты одной командой
test_all: $(filter test_%, $(BIN))
	@echo "Running test_rle:"
	./test_rle
	@echo ""
	@echo "Running test_lz77:"
	./test_lz77
	@echo ""
	@echo "Running test_lzw:"
	./test_lzw
	@echo ""
	@echo "Running test_huffman:"
	./test_huffman
	@echo ""
	@echo "Running test_bwt:"
	./test_bwt
	@echo ""
	@echo "Running test_bwt_rle_huffman:"
	./test_bwt_rle_huffman
	@echo ""
	@echo "Running test_rle_cuda:"
	./test_rle_cuda
	@echo ""
	@echo "Running test_bwt_cuda:"
	./test_bwt_cuda
	@echo ""
	@echo "Running test_huffman_cuda:"
	./test_huffman_cuda
