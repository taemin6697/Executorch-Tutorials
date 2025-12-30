#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::from_blob;
using ::executorch::runtime::Error;

struct Stats {
  double avg_ms{0.0};
  double min_ms{0.0};
  double max_ms{0.0};
  double p50_ms{0.0};
  double p95_ms{0.0};
  double stddev_ms{0.0};
};

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0
      << " [gpu_pte] [cpu_pte] [warmup_iters] [iters] [H] [W]\n\n"
      << "Defaults:\n"
      << "  gpu_pte      = ./model_gpu.pte\n"
      << "  cpu_pte      = ./model_cpu.pte\n"
      << "  warmup_iters = 5\n"
      << "  iters        = 30\n"
      << "  H, W         = 224, 224 (input shape = [1, 3, H, W])\n";
}

static Stats compute_stats(std::vector<double> samples_ms) {
  Stats s;
  if (samples_ms.empty()) {
    return s;
  }

  const double sum =
      std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
  s.avg_ms = sum / static_cast<double>(samples_ms.size());

  auto [mn_it, mx_it] = std::minmax_element(samples_ms.begin(), samples_ms.end());
  s.min_ms = *mn_it;
  s.max_ms = *mx_it;

  std::sort(samples_ms.begin(), samples_ms.end());
  auto percentile = [&](double p) -> double {
    if (samples_ms.size() == 1) {
      return samples_ms[0];
    }
    const double idx = p * static_cast<double>(samples_ms.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double t = idx - static_cast<double>(lo);
    return samples_ms[lo] * (1.0 - t) + samples_ms[hi] * t;
  };
  s.p50_ms = percentile(0.50);
  s.p95_ms = percentile(0.95);

  double var = 0.0;
  for (double x : samples_ms) {
    const double d = x - s.avg_ms;
    var += d * d;
  }
  var /= static_cast<double>(samples_ms.size());
  s.stddev_ms = std::sqrt(var);
  return s;
}

static Stats run_benchmark(
    Module& module,
    const char* name,
    const TensorPtr& input_tensor,
    int warmup_iters,
    int iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    const auto r = module.forward(input_tensor);
    if (!r.ok()) {
      std::cerr << name << " warmup failed. Error=" << static_cast<int>(r.error())
                << std::endl;
      return {};
    }
  }

  std::vector<double> times_ms;
  times_ms.reserve(static_cast<size_t>(iters));

  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = module.forward(input_tensor);
    const auto t1 = std::chrono::steady_clock::now();
    if (!r.ok()) {
      std::cerr << name << " failed. Error=" << static_cast<int>(r.error())
                << std::endl;
      return {};
    }
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    times_ms.push_back(ms);
  }

  return compute_stats(std::move(times_ms));
}

static void print_stats(const char* label, const Stats& s) {
  std::cout << label << " avg/min/p50/p95/max/std (ms): " << s.avg_ms << " / "
            << s.min_ms << " / " << s.p50_ms << " / " << s.p95_ms << " / "
            << s.max_ms << " / " << s.stddev_ms << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc >= 2 && (std::strcmp(argv[1], "-h") == 0 ||
                    std::strcmp(argv[1], "--help") == 0)) {
    print_usage(argv[0]);
    return 0;
  }

  const char* gpu_path = (argc >= 2) ? argv[1] : "./model_gpu.pte";
  const char* cpu_path = (argc >= 3) ? argv[2] : "./model_cpu.pte";
  const int warmup_iters = (argc >= 4) ? std::atoi(argv[3]) : 5;
  const int iters = (argc >= 5) ? std::atoi(argv[4]) : 30;
  const int H = (argc >= 6) ? std::atoi(argv[5]) : 224;
  const int W = (argc >= 7) ? std::atoi(argv[6]) : 224;

  if (warmup_iters < 0 || iters <= 0 || H <= 0 || W <= 0) {
    print_usage(argv[0]);
    return 2;
  }

  std::cout << "GPU PTE: " << gpu_path << std::endl;
  std::cout << "CPU PTE: " << cpu_path << std::endl;
  std::cout << "Warmup: " << warmup_iters << ", Iters: " << iters << std::endl;
  std::cout << "Input: [1, 3, " << H << ", " << W << "] float32" << std::endl;

  Module module_gpu(gpu_path);
  Module module_cpu(cpu_path);

  const auto gpu_load_err = module_gpu.load();
  if (gpu_load_err != Error::Ok) {
    std::cerr << "GPU module load failed. Error=" << static_cast<int>(gpu_load_err)
              << std::endl;
    return 1;
  }
  const auto cpu_load_err = module_cpu.load();
  if (cpu_load_err != Error::Ok) {
    std::cerr << "CPU module load failed. Error=" << static_cast<int>(cpu_load_err)
              << std::endl;
    return 2;
  }

  std::vector<float> input(static_cast<size_t>(1 * 3 * H * W), 0.0f);
  TensorPtr tensor = from_blob(input.data(), {1, 3, H, W});

  const Stats gpu_stats =
      run_benchmark(module_gpu, "GPU", tensor, warmup_iters, iters);
  const Stats cpu_stats =
      run_benchmark(module_cpu, "CPU", tensor, warmup_iters, iters);

  print_stats("GPU", gpu_stats);
  print_stats("CPU", cpu_stats);

  if (gpu_stats.avg_ms > 0.0 && cpu_stats.avg_ms > 0.0) {
    std::cout << "Speedup (CPU/GPU, avg): " << (cpu_stats.avg_ms / gpu_stats.avg_ms)
              << "x" << std::endl;
  }

  return 0;
}


