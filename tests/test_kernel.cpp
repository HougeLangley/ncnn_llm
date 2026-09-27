#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <net.h>
#include <benchmark.h>
#include <allocator.h>

#include "kernel/gdr.h"
#include "kernel/shortconv.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include "kernel/x86/gdr_x86.h"
#include "kernel/x86/shortconv_x86.h"
#endif


using namespace ncnn;

class TestNet : public ncnn::Net
{
public:
    using ncnn::Net::create_custom_layer;
};

class TrackingAllocator : public ncnn::Allocator
{
public:
    TrackingAllocator() : malloc_count(0), free_count(0), bytes_allocated(0) {}
    virtual void* fastMalloc(size_t size)
    {
        malloc_count++;
        bytes_allocated += size;
        return ncnn::fastMalloc(size);
    }
    virtual void fastFree(void* ptr)
    {
        if (ptr)
        {
            free_count++;
            ncnn::fastFree(ptr);
        }
    }
    int malloc_count;
    int free_count;
    size_t bytes_allocated;
};

static float rand_float(float val_min = -1.f, float val_max = 1.f)
{
    float r = (float)rand() / (float)RAND_MAX;
    return val_min + r * (val_max - val_min);
}

static void fill_mat(ncnn::Mat& m)
{
    float* p = (float*)m.data;
    for (size_t i = 0; i < m.total(); i++)
    {
        p[i] = rand_float();
    }
}

static float max_abs_diff(const ncnn::Mat& a, const ncnn::Mat& b)
{
    float max_diff = 0.f;
    const float* pa = (const float*)a.data;
    const float* pb = (const float*)b.data;
    size_t total = a.total();
    for (size_t i = 0; i < total; i++)
    {
        float diff = std::fabs(pa[i] - pb[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

int main()
{
    srand(42);

    ncnn::Option opt;
    opt.num_threads = 4;

    std::cout << "===== Custom Operator Dispatch & x86 Optimization Tests =====" << std::endl;

    // 1. Test GatedDeltaRule with seq_len = 1 and seq_len = 4
    for (int seq_len : {1, 4})
    {
        std::cout << "\nTesting GatedDeltaRule (seq_len=" << seq_len << "):" << std::endl;
        int num_heads = 16;
        int k_head_dim = 64;
        int v_head_dim = 64;

        ncnn::Mat A_log(num_heads);
        ncnn::Mat dt_bias(num_heads);
        ncnn::Mat b(num_heads, seq_len);
        ncnn::Mat a(num_heads, seq_len);
        ncnn::Mat query(k_head_dim, num_heads, seq_len);
        ncnn::Mat key(k_head_dim, num_heads, seq_len);
        ncnn::Mat value(v_head_dim, num_heads, seq_len);
        ncnn::Mat initial_state(v_head_dim, k_head_dim, num_heads);

        fill_mat(A_log);
        fill_mat(dt_bias);
        fill_mat(b);
        fill_mat(a);
        fill_mat(query);
        fill_mat(key);
        fill_mat(value);
        fill_mat(initial_state);

        std::vector<ncnn::Mat> bottom_blobs = {A_log, dt_bias, b, a, query, key, value, initial_state};

        GatedDeltaRule naive_gdr;
        std::vector<ncnn::Mat> top_naive(2);
        int ret_naive = naive_gdr.forward(bottom_blobs, top_naive, opt);
        if (ret_naive != 0)
        {
            std::cerr << "Naive GDR forward failed: " << ret_naive << std::endl;
            return 1;
        }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        GatedDeltaRule_x86 x86_gdr;
        std::vector<ncnn::Mat> top_x86(2);
        int ret_x86 = x86_gdr.forward(bottom_blobs, top_x86, opt);
        if (ret_x86 != 0)
        {
            std::cerr << "x86 GDR forward failed: " << ret_x86 << std::endl;
            return 1;
        }

        float diff_out = max_abs_diff(top_naive[0], top_x86[0]);
        float diff_state = max_abs_diff(top_naive[1], top_x86[1]);

        std::cout << "  Diff attn output: " << diff_out << std::endl;
        std::cout << "  Diff state output: " << diff_state << std::endl;

        if (diff_out > 1e-4f || diff_state > 1e-4f)
        {
            std::cerr << "FAILED: GDR x86 output differs too much from naive!" << std::endl;
            return 1;
        }
        std::cout << "  GDR accuracy test PASSED!" << std::endl;

        // Simple benchmark
        const int iters = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++)
        {
            naive_gdr.forward(bottom_blobs, top_naive, opt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++)
        {
            x86_gdr.forward(bottom_blobs, top_x86, opt);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double x86_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
        std::cout << "  Benchmark: naive = " << naive_ms << " ms, x86 = " << x86_ms << " ms (speedup: " << naive_ms / x86_ms << "x)" << std::endl;
#endif
    }

    // 2. Test ShortConv with seq_len = 1 and seq_len = 4
    for (int seq_len : {1, 4})
    {
        std::cout << "\nTesting ShortConv (seq_len=" << seq_len << "):" << std::endl;
        int groups = 2048;
        int kernel_size = 4;

        ncnn::Mat weight_mat(kernel_size, 1, groups);
        ncnn::Mat mixed_qkv(groups, seq_len);
        ncnn::Mat conv_state(groups, kernel_size);

        fill_mat(weight_mat);
        fill_mat(mixed_qkv);
        fill_mat(conv_state);

        std::vector<ncnn::Mat> bottom_blobs = {weight_mat, mixed_qkv, conv_state};

        ShortConv naive_sc;
        std::vector<ncnn::Mat> top_naive(2);
        int ret_naive = naive_sc.forward(bottom_blobs, top_naive, opt);
        if (ret_naive != 0)
        {
            std::cerr << "Naive ShortConv forward failed: " << ret_naive << std::endl;
            return 1;
        }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        ShortConv_x86 x86_sc;
        std::vector<ncnn::Mat> top_x86(2);
        int ret_x86 = x86_sc.forward(bottom_blobs, top_x86, opt);
        if (ret_x86 != 0)
        {
            std::cerr << "x86 ShortConv forward failed: " << ret_x86 << std::endl;
            return 1;
        }

        float diff_out = max_abs_diff(top_naive[0], top_x86[0]);
        float diff_state = max_abs_diff(top_naive[1], top_x86[1]);

        std::cout << "  Diff output: " << diff_out << std::endl;
        std::cout << "  Diff state output: " << diff_state << std::endl;

        if (diff_out > 1e-5f || diff_state > 1e-5f)
        {
            std::cerr << "FAILED: ShortConv x86 output differs too much from naive!" << std::endl;
            return 1;
        }
        std::cout << "  ShortConv accuracy test PASSED!" << std::endl;

        const int iters = 200;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++)
        {
            naive_sc.forward(bottom_blobs, top_naive, opt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++)
        {
            x86_sc.forward(bottom_blobs, top_x86, opt);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double x86_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
        std::cout << "  Benchmark: naive = " << naive_ms << " ms, x86 = " << x86_ms << " ms (speedup: " << naive_ms / x86_ms << "x)" << std::endl;
#endif
    }

    // 3. Test Net custom layer registration and automatic dispatch
    std::cout << "\nTesting automatic Net layer registration dispatch:" << std::endl;
    TestNet test_net;
    register_gdr_layers(test_net);

    ncnn::Layer* gdr_layer = test_net.create_custom_layer("GatedDeltaRule");
    ncnn::Layer* sc_layer = test_net.create_custom_layer("ShortConv");

    if (!gdr_layer)
    {
        std::cerr << "FAILED: create_custom_layer(GatedDeltaRule) returned null!" << std::endl;
        return 1;
    }
    if (!sc_layer)
    {
        std::cerr << "FAILED: create_custom_layer(ShortConv) returned null!" << std::endl;
        return 1;
    }

    std::cout << "  Created GatedDeltaRule layer: " << typeid(*gdr_layer).name() << std::endl;
    std::cout << "  Created ShortConv layer: " << typeid(*sc_layer).name() << std::endl;

    delete gdr_layer;
    delete sc_layer;

    // 4. Test Memory Pool Allocation (opt.blob_allocator & opt.workspace_allocator)
    std::cout << "\nTesting Memory Pool Allocation (TrackingAllocator & PoolAllocator):" << std::endl;
    {
        TrackingAllocator blob_allocator;
        TrackingAllocator workspace_allocator;

        ncnn::Option pool_opt;
        pool_opt.num_threads = 4;
        pool_opt.blob_allocator = &blob_allocator;
        pool_opt.workspace_allocator = &workspace_allocator;

        int seq_len = 4;
        int num_heads = 16;
        int k_head_dim = 64;
        int v_head_dim = 64;

        ncnn::Mat A_log(num_heads);
        ncnn::Mat dt_bias(num_heads);
        ncnn::Mat b(num_heads, seq_len);
        ncnn::Mat a(num_heads, seq_len);
        ncnn::Mat query(k_head_dim, num_heads, seq_len);
        ncnn::Mat key(k_head_dim, num_heads, seq_len);
        ncnn::Mat value(v_head_dim, num_heads, seq_len);
        ncnn::Mat initial_state(v_head_dim, k_head_dim, num_heads);

        fill_mat(A_log);
        fill_mat(dt_bias);
        fill_mat(b);
        fill_mat(a);
        fill_mat(query);
        fill_mat(key);
        fill_mat(value);
        fill_mat(initial_state);

        std::vector<ncnn::Mat> bottom_blobs = {A_log, dt_bias, b, a, query, key, value, initial_state};

        // Test Naive GDR with allocators
        GatedDeltaRule naive_gdr;
        std::vector<ncnn::Mat> top_naive(2);
        int ret = naive_gdr.forward(bottom_blobs, top_naive, pool_opt);
        if (ret != 0 || top_naive[0].allocator != &blob_allocator || top_naive[1].allocator != &blob_allocator)
        {
            std::cerr << "FAILED: GDR naive blob allocator check failed!" << std::endl;
            return 1;
        }
        if (workspace_allocator.malloc_count == 0 || workspace_allocator.malloc_count != workspace_allocator.free_count)
        {
            std::cerr << "FAILED: GDR naive workspace allocator leak or no-use detected!" << std::endl;
            return 1;
        }

        // Test x86 GDR with allocators
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        workspace_allocator.malloc_count = 0;
        workspace_allocator.free_count = 0;
        GatedDeltaRule_x86 x86_gdr;
        std::vector<ncnn::Mat> top_x86(2);
        ret = x86_gdr.forward(bottom_blobs, top_x86, pool_opt);
        if (ret != 0 || top_x86[0].allocator != &blob_allocator || top_x86[1].allocator != &blob_allocator)
        {
            std::cerr << "FAILED: GDR x86 blob allocator check failed!" << std::endl;
            return 1;
        }
        if (workspace_allocator.malloc_count == 0 || workspace_allocator.malloc_count != workspace_allocator.free_count)
        {
            std::cerr << "FAILED: GDR x86 workspace allocator leak or no-use detected!" << std::endl;
            return 1;
        }
#endif

        // Test ShortConv with allocators
        int groups = 2048;
        int kernel_size = 4;
        ncnn::Mat weight_mat(kernel_size, 1, groups);
        ncnn::Mat mixed_qkv(groups, seq_len);
        ncnn::Mat conv_state(groups, kernel_size);
        fill_mat(weight_mat);
        fill_mat(mixed_qkv);
        fill_mat(conv_state);

        std::vector<ncnn::Mat> sc_bottoms = {weight_mat, mixed_qkv, conv_state};

        workspace_allocator.malloc_count = 0;
        workspace_allocator.free_count = 0;
        ShortConv naive_sc;
        std::vector<ncnn::Mat> sc_top_naive(2);
        ret = naive_sc.forward(sc_bottoms, sc_top_naive, pool_opt);
        if (ret != 0 || sc_top_naive[0].allocator != &blob_allocator || sc_top_naive[1].allocator != &blob_allocator)
        {
            std::cerr << "FAILED: ShortConv naive blob allocator check failed!" << std::endl;
            return 1;
        }
        if (workspace_allocator.malloc_count == 0 || workspace_allocator.malloc_count != workspace_allocator.free_count)
        {
            std::cerr << "FAILED: ShortConv naive workspace allocator leak or no-use detected!" << std::endl;
            return 1;
        }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        workspace_allocator.malloc_count = 0;
        workspace_allocator.free_count = 0;
        ShortConv_x86 x86_sc;
        std::vector<ncnn::Mat> sc_top_x86(2);
        ret = x86_sc.forward(sc_bottoms, sc_top_x86, pool_opt);
        if (ret != 0 || sc_top_x86[0].allocator != &blob_allocator || sc_top_x86[1].allocator != &blob_allocator)
        {
            std::cerr << "FAILED: ShortConv x86 blob allocator check failed!" << std::endl;
            return 1;
        }
        if (workspace_allocator.malloc_count == 0 || workspace_allocator.malloc_count != workspace_allocator.free_count)
        {
            std::cerr << "FAILED: ShortConv x86 workspace allocator leak or no-use detected!" << std::endl;
            return 1;
        }
#endif
        std::cout << "  Memory pool allocation & deallocation tracking tests PASSED!" << std::endl;
    }

    std::cout << "\nALL TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
