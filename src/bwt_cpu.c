#include <bwt.h>

static void radix_pass(int *a, int *b, int *r, int n, int K) {
    int *cnt = (int *)calloc(K + 1, sizeof(int));
    for (int i = 0; i < n; i++) cnt[r[a[i]]]++;
    for (int i = 1; i <= K; i++) cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--) b[--cnt[r[a[i]]]] = a[i];
    free(cnt);
}

void build_suffix_array(const char *s, int n, int *sa) {
    int *rank = (int *)malloc(n * sizeof(int));
    int *temp = (int *)malloc(n * sizeof(int));
    int *sa2 = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        sa[i] = i;
        rank[i] = (unsigned char)s[i];
    }

    for (int k = 1; k < n; k <<= 1) {
        for (int i = 0; i < n; i++)
            temp[i] = (i + k < n) ? rank[i + k] + 1 : 0;
        radix_pass(sa, sa2, temp, n, n + 1);
        for (int i = 0; i < n; i++)
            temp[i] = rank[i] + 1;
        radix_pass(sa2, sa, temp, n, n + 1);
        temp[sa[0]] = 0;
        for (int i = 1; i < n; i++) {
            int prev = sa[i - 1], curr = sa[i];
            int same = rank[prev] == rank[curr] &&
                       ((prev + k < n && curr + k < n) ?
                        rank[prev + k] == rank[curr + k] :
                        (prev + k >= n && curr + k >= n));
            temp[curr] = temp[prev] + !same;
        }
        memcpy(rank, temp, n * sizeof(int));
        if (rank[sa[n - 1]] == n - 1) break;
    }

    free(rank);
    free(temp);
    free(sa2);
}

void bwt_transform_cpu(const char *input_file, const char *output_file) {
    FILE *f = fopen(input_file, "rb");
    if (!f) { perror("fopen"); return; }

    fseek(f, 0, SEEK_END);
    int n0 = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *s = (char *)malloc(n0);
    fread(s, 1, n0, f);
    fclose(f);

    unsigned char used[256] = {0};
    for (int i = 0; i < n0; i++) used[(unsigned char)s[i]] = 1;
    unsigned char sentinel = 0;
    while (used[sentinel] && sentinel < 255) sentinel++;
    if (used[sentinel]) {
        fprintf(stderr, "Cannot find unique terminator byte\n");
        exit(1);
    }

    int n = n0 + 1;
    char *s_term = (char *)malloc(n);
    memcpy(s_term, s, n0);
    s_term[n0] = sentinel;

    clock_t start = clock();

    int *sa = (int *)malloc(n * sizeof(int));
    build_suffix_array(s_term, n, sa);

    char *bwt = (char *)malloc(n);
    int primary_index = 0;
    for (int i = 0; i < n; ++i) {
        if (sa[i] == 0) {
            bwt[i] = s_term[n - 1];
            primary_index = i;
        } else {
            bwt[i] = s_term[sa[i] - 1];
        }
    }

    clock_t end = clock();
    printf("%s BWT CPU transform time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    FILE *out = fopen(output_file, "wb");
    fwrite(&primary_index, sizeof(int), 1, out);
    fwrite(&sentinel, sizeof(unsigned char), 1, out);
    fwrite(bwt, 1, n, out);
    fclose(out);

    free(s);
    free(s_term);
    free(bwt);
    free(sa);
}

void bwt_inverse_cpu(const char *input_file, const char *output_file) {
    FILE *f = fopen(input_file, "rb");
    if (!f) { perror("fopen"); return; }

    fseek(f, 0, SEEK_END);
    int n = ftell(f) - sizeof(int) - sizeof(unsigned char);
    fseek(f, 0, SEEK_SET);

    int primary_index;
    unsigned char sentinel;
    fread(&primary_index, sizeof(int), 1, f);
    fread(&sentinel, sizeof(unsigned char), 1, f);

    char *last_col = (char *)malloc(n);
    fread(last_col, 1, n, f);
    fclose(f);

    clock_t start = clock();

    int count[256] = {0};
    for (int i = 0; i < n; ++i)
        count[(unsigned char)last_col[i]]++;

    int total[256];
    int sum = 0;
    for (int i = 0; i < 256; ++i) {
        total[i] = sum;
        sum += count[i];
    }

    int temp[256];
    memcpy(temp, total, sizeof(total));

    int *next = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) {
        int c = (unsigned char)last_col[i];
        next[temp[c]++] = i;
    }

    char *output = (char *)malloc(n);
    int idx = primary_index;
    for (int i = 0; i < n; ++i) {
        output[i] = last_col[idx];
        idx = next[idx];
    }

    clock_t end = clock();
    printf("%s BWT CPU inverse time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Убираем sentinel
    FILE *out = fopen(output_file, "wb");
    for (int i = 0; i < n; ++i) {
        if ((unsigned char)output[i] != sentinel)
            fwrite(&output[i], 1, 1, out);
    }
    fclose(out);

    free(last_col);
    free(output);
    free(next);
}
