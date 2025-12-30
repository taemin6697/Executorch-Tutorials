#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <executorch/runtime/core/error.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace ::executorch::extension;
using ::executorch::runtime::Error;

int main(int argc, char* argv[]) {
    const char* model_path = (argc >= 2) ? argv[1] : "./model.pte";
    const int warmup_iters = (argc >= 3) ? std::atoi(argv[2]) : 5;
    const int iters = (argc >= 4) ? std::atoi(argv[3]) : 50;

    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Warmup: " << warmup_iters << ", Iters: " << iters << std::endl;

    // Load the model.
    Module module(model_path);
    const auto load_error = module.load();
    if (load_error != Error::Ok) {
        std::cerr << "Failed to load model. Error=" << static_cast<int>(load_error)
                  << std::endl;
        return 1;
    }

    // Create an input tensor.
    std::vector<float> input(1 * 3 * 224 * 224, 0.0f);
    auto tensor = from_blob(input.data(), {1, 3, 224, 224});

    // Warmup.
    for (int i = 0; i < warmup_iters; ++i) {
        const auto r = module.forward(tensor);
        if (!r.ok()) {
            std::cerr << "Warmup failed. Error=" << static_cast<int>(r.error())
                      << std::endl;
            return 2;
        }
    }

    // Benchmark.
    double sum_ms = 0.0;
    double min_ms = 1e30;
    double max_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = module.forward(tensor);
        const auto t1 = std::chrono::steady_clock::now();

        if (!r.ok()) {
            std::cerr << "Inference failed. Error=" << static_cast<int>(r.error())
                      << std::endl;
            return 3;
        }

        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    std::cout << "ExecuTorch forward avg/min/max (ms): "
              << (sum_ms / static_cast<double>(iters)) << " / " << min_ms
              << " / " << max_ms << std::endl;

        std::cout << "Success" << std::endl;
    return 0;
}