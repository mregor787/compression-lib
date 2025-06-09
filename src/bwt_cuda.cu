#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "bwt.h"

struct rank_pair {
    int first, second;

    __host__ __device__
    bool operator<(const rank_pair& other) const {
        return (first < other.first) || (first == other.first && second < other.second);
    }

    __host__ __device__
    bool operator==(const rank_pair& other) const {
        return first == other.first && second == other.second;
    }
};

struct make_rank_pair {
    const int* rank;
    int n, k;

    make_rank_pair(const int* r, int len, int offset)
        : rank(r), n(len), k(offset) {}

    __host__ __device__
    rank_pair operator()(const int& i) const {
        return {
            rank[i],
            (i + k < n) ? rank[i + k] : -1
        };
    }
};

struct is_different {
    __host__ __device__
    int operator()(const rank_pair& a, const rank_pair& b) const {
        return !(a == b);
    }
};

void build_suffix_array_gpu(const thrust::device_vector<unsigned char>& d_text, thrust::device_vector<int>& d_sa) {
    int n = d_text.size();

    thrust::device_vector<int> rank(n);
    thrust::copy(d_text.begin(), d_text.end(), rank.begin());

    thrust::sequence(d_sa.begin(), d_sa.end());
    thrust::device_vector<rank_pair> d_key(n);
    thrust::device_vector<int> flags(n);

    for (int k = 1; k < n; k <<= 1) {
        make_rank_pair pair_gen(thrust::raw_pointer_cast(rank.data()), n, k);
        thrust::transform(
            d_sa.begin(), d_sa.end(),
            d_key.begin(),
            pair_gen
        );

        thrust::sort_by_key(
            d_key.begin(), d_key.end(),
            d_sa.begin()
        );

        thrust::transform(
            d_key.begin() + 1, d_key.end(),
            d_key.begin(),
            flags.begin() + 1,
            is_different()
        );
        flags[0] = 0;

        thrust::inclusive_scan(flags.begin(), flags.end(), flags.begin());

        thrust::scatter(flags.begin(), flags.end(), d_sa.begin(), rank.begin());

        if (flags[n - 1] == n - 1) break;
    }
}

void bwt_transform_cuda(const char* input_file, const char* output_file) {
    FILE* f = fopen(input_file, "rb");
    if (!f) {
        perror("fopen input");
        return;
    }

    fseek(f, 0, SEEK_END);
    int n0 = ftell(f);
    fseek(f, 0, SEEK_SET);

    unsigned char* s = (unsigned char*)malloc(n0 + 1);
    fread(s, 1, n0, f);
    fclose(f);

    unsigned char used[256] = {0};
    for (int i = 0; i < n0; i++) used[s[i]] = 1;
    unsigned char sentinel = 0;
    while (used[sentinel] && sentinel < 255) sentinel++;
    if (used[sentinel]) {
        fprintf(stderr, "No unique sentinel byte found\n");
        exit(1);
    }
    int n = n0 + 1;
    s[n0] = sentinel;

    thrust::device_vector<unsigned char> d_text(s, s + n);

    clock_t start = clock();

    thrust::device_vector<int> d_sa(n);
    build_suffix_array_gpu(d_text, d_sa);

    thrust::host_vector<int> h_sa = d_sa;
    thrust::host_vector<unsigned char> h_text = d_text;

    unsigned char* bwt_result = (unsigned char*)malloc(n);
    int primary_index = 0;
    for (int i = 0; i < n; ++i) {
        int idx = h_sa[i];
        bwt_result[i] = h_text[(idx + n - 1) % n];
        if (idx == 0) primary_index = i;
    }

    clock_t end = clock();
    printf("%s BWT GPU transform time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE* out = fopen(output_file, "wb");
    if (!out) {
        perror("fopen output");
        free(s);
        free(bwt_result);
        return;
    }

    fwrite(&primary_index, sizeof(int), 1, out);
    fwrite(&sentinel, sizeof(unsigned char), 1, out);
    fwrite(bwt_result, 1, n, out);
    fclose(out);

    free(s);
    free(bwt_result);
}

void bwt_inverse_cuda(const char *input_file, const char *output_file) {
    FILE *in = fopen(input_file, "rb");
    if (!in) {
        perror("fopen input");
        return;
    }

    int primary_index;
    unsigned char sentinel;

    fread(&primary_index, sizeof(int), 1, in);
    fread(&sentinel, sizeof(unsigned char), 1, in);

    fseek(in, 0, SEEK_END);
    long file_size = ftell(in);
    size_t n = file_size - sizeof(int) - sizeof(unsigned char);
    fseek(in, sizeof(int) + sizeof(unsigned char), SEEK_SET);

    unsigned char *bwt_host = (unsigned char *)malloc(n);
    fread(bwt_host, 1, n, in);
    fclose(in);

    thrust::device_vector<unsigned char> d_bwt(bwt_host, bwt_host + n);

    clock_t start = clock();

    thrust::device_vector<int> d_indices(n);
    thrust::sequence(d_indices.begin(), d_indices.end());

    thrust::stable_sort_by_key(
        thrust::device,
        d_bwt.begin(),
        d_bwt.end(),
        d_indices.begin()
    );

    thrust::device_vector<int> d_next(n);
    thrust::scatter(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_indices.begin(),
        d_next.begin()
    );

    int *next = (int *)malloc(n * sizeof(int));
    thrust::copy(d_next.begin(), d_next.end(), next);

    unsigned char *output = (unsigned char *)malloc(n);
    int idx = primary_index;
    for (size_t i = 0; i < n; ++i) {
        output[n - 1 - i] = bwt_host[idx];
        idx = next[idx];
    }

    clock_t end = clock();
    printf("%s BWT GPU inverse time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE *out = fopen(output_file, "wb");
    if (!out) {
        perror("fopen output");
        free(bwt_host);
        free(output);
        free(next);
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        if (output[i] != sentinel) {
            fwrite(&output[i], 1, 1, out);
        }
    }

    fclose(out);

    free(bwt_host);
    free(output);
    free(next);
}