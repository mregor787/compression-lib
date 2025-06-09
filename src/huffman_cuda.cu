#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <huffman.h>

static Node* new_node(int byte, uint32_t freq, Node* left, Node* right) {
    Node* n = (Node*)malloc(sizeof(Node));
    n->byte = byte;
    n->freq = freq;
    n->left = left;
    n->right = right;
    return n;
}

static void heap_insert(MinHeap *h, Node *n) {
    int i = h->size++;
    while (i && n->freq < h->data[(i - 1) / 2]->freq) {
        h->data[i] = h->data[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    h->data[i] = n;
}

static Node* heap_extract(MinHeap *h) {
    Node* min = h->data[0];
    Node* last = h->data[--h->size];
    int i = 0;
    while ((2 * i + 1) < h->size) {
        int smallest = 2 * i + 1;
        if (smallest + 1 < h->size && h->data[smallest + 1]->freq < h->data[smallest]->freq)
            smallest++;
        if (last->freq <= h->data[smallest]->freq)
            break;
        h->data[i] = h->data[smallest];
        i = smallest;
    }
    h->data[i] = last;
    return min;
}

static void build_codes(Node* root, HuffCode* table, uint64_t code, int depth) {
    if (!root->left && !root->right) {
        table[root->byte].bits = code;
        table[root->byte].length = depth;
        return;
    }
    if (root->left) build_codes(root->left, table, (code << 1), depth + 1);
    if (root->right) build_codes(root->right, table, (code << 1) | 1, depth + 1);
}

static void free_tree(Node* root) {
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

void compute_frequencies_gpu(const unsigned char* input, size_t size, uint32_t* freq_out) {
    thrust::device_vector<unsigned char> d_input(input, input + size);

    thrust::sort(d_input.begin(), d_input.end());

    thrust::device_vector<unsigned char> d_keys(BYTE_RANGE);
    thrust::device_vector<uint32_t> d_counts(BYTE_RANGE);

    auto end = thrust::reduce_by_key(
        d_input.begin(), d_input.end(), 
        thrust::make_constant_iterator(1),  
        d_keys.begin(), 
        d_counts.begin() 
    );

    size_t n = end.first - d_keys.begin();

    thrust::host_vector<uint32_t> h_freq(BYTE_RANGE, 0);

    thrust::host_vector<unsigned char> h_keys(d_keys.begin(), d_keys.begin() + n);
    thrust::host_vector<uint32_t> h_counts(d_counts.begin(), d_counts.begin() + n);

    for (size_t i = 0; i < n; i++) {
        h_freq[h_keys[i]] = h_counts[i];
    }

    memcpy(freq_out, h_freq.data(), BYTE_RANGE * sizeof(uint32_t));
}

void huffman_compress_cuda(const char* input_file, const char* output_file) {
    FILE* in = fopen(input_file, "rb");
    if (!in) { perror("fopen"); return; }

    fseek(in, 0, SEEK_END);
    long input_size = ftell(in);
    fseek(in, 0, SEEK_SET);

    unsigned char* input = (unsigned char*)malloc(input_size);
    fread(input, 1, input_size, in);
    fclose(in);

    clock_t start = clock();

    uint32_t freq[BYTE_RANGE] = {0};
    compute_frequencies_gpu(input, input_size, freq);

    clock_t mid = clock();
    printf("%s Huffman GPU Freq Count time: %.2f ms\n", input_file, 1000.0 * (mid - start) / CLOCKS_PER_SEC);

    MinHeap heap = { .data = (Node**)malloc(BYTE_RANGE * sizeof(Node*)), .size = 0 };
    for (int i = 0; i < BYTE_RANGE; i++) {
        if (freq[i]) heap_insert(&heap, new_node(i, freq[i], NULL, NULL));
    }

    while (heap.size > 1) {
        Node* l = heap_extract(&heap);
        Node* r = heap_extract(&heap);
        heap_insert(&heap, new_node(-1, l->freq + r->freq, l, r));
    }

    Node* root = heap_extract(&heap);
    HuffCode table[BYTE_RANGE] = {0};
    build_codes(root, table, 0, 0);

    FILE* out = fopen(output_file, "wb");
    if (!out) { perror("fopen"); return; }

    fwrite(freq, sizeof(uint32_t), BYTE_RANGE, out);
    fwrite(&input_size, sizeof(uint32_t), 1, out);

    uint8_t buffer = 0;
    int bit_count = 0;
    for (long i = 0; i < input_size; i++) {
        HuffCode code = table[input[i]];
        for (int b = code.length - 1; b >= 0; b--) {
            buffer <<= 1;
            buffer |= (code.bits >> b) & 1;
            bit_count++;
            if (bit_count == 8) {
                fwrite(&buffer, 1, 1, out);
                buffer = 0;
                bit_count = 0;
            }
        }
    }
    if (bit_count > 0) {
        buffer <<= (8 - bit_count);
        fwrite(&buffer, 1, 1, out);
    }

    clock_t end = clock();
    printf("%s Huffman GPU compress time: %.2f ms\n", input_file, 1000.0 * (end - start) / CLOCKS_PER_SEC);

    free_tree(root);
    fclose(out);
    free(input);
}