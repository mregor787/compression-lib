#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <cuda_runtime.h>
#include "bwt.h"

// Компаратор циклических сдвигов
struct BWTComparator {
    const uint8_t* data;
    size_t n;

    BWTComparator(const uint8_t* _data, size_t _n) : data(_data), n(_n) {}

    __host__ __device__
    bool operator()(const int& a, const int& b) const {
        for (size_t i = 0; i < n; ++i) {
            uint8_t ca = data[(a + i) % n];
            uint8_t cb = data[(b + i) % n];
            if (ca < cb) return true;
            if (ca > cb) return false;
        }
        return false;
    }
};

int bwt_transform_cuda(const uint8_t *input, size_t n,
                       uint8_t *output, size_t *primary_index) {
    if (!input || !output || !primary_index || n == 0)
        return -1;
        
    // Загрузка входных данных на устройство
    thrust::device_vector<uint8_t> d_input(input, input + n);
    thrust::device_vector<int> indices(n);
    thrust::sequence(indices.begin(), indices.end());

    // Компаратор по циклическим сдвигам
    uint8_t* d_input_ptr = thrust::raw_pointer_cast(d_input.data());
    BWTComparator comp(d_input_ptr, n);

    // Сортировка индексов по циклическим сдвигам
    thrust::sort(thrust::device, indices.begin(), indices.end(), comp);

    // Формируем BWT-выход — последний столбец
    thrust::device_vector<uint8_t> d_output(n);
    thrust::transform(
        indices.begin(), indices.end(), d_output.begin(),
        [=] __device__ (int idx) {
            return d_input_ptr[(idx + n - 1) % n];
        });

    // Поиск позиции исходной строки (смещение == 0)
    thrust::host_vector<int> h_indices = indices;
    for (size_t i = 0; i < n; ++i) {
        if (h_indices[i] == 0) {
            *primary_index = i;
            break;
        }
    }

    // Копируем результат обратно
    thrust::copy(d_output.begin(), d_output.end(), output);

    return 0;
}

int bwt_inverse_cuda(const uint8_t *input, size_t n,
                     uint8_t *output, size_t primary_index) {
    if (!input || !output || n == 0 || primary_index >= n)
        return -1;

    // Последний столбец (вектор входных байтов)
    thrust::device_vector<uint8_t> last_column(input, input + n);
    thrust::device_vector<uint8_t> first_column = last_column;

    // Индексы [0, 1, 2, ..., n-1]
    thrust::device_vector<int> original_pos(n);
    thrust::sequence(original_pos.begin(), original_pos.end());

    // Сортировка first_column с сохранением информации об исходных позициях
    thrust::sort_by_key(first_column.begin(), first_column.end(), original_pos.begin());

    // Таблица переходов LF-мэппинга
    thrust::device_vector<int> T(n);
    thrust::scatter(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        original_pos.begin(),  // ключи = где были
        T.begin()              // значения = куда идут
    );

    // Копируем на host для восстановления
    thrust::host_vector<uint8_t> h_last = last_column;
    thrust::host_vector<int> h_T = T;

    size_t idx = primary_index;
    for (size_t i = 0; i < n; ++i) {
        output[n - 1 - i] = h_last[idx];
        idx = h_T[idx];
    }

    return 0;
}
