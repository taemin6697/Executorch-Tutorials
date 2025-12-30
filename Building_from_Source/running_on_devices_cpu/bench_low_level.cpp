#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#
#include <executorch/runtime/executor/memory_manager.h>
#
using executorch::extension::FileDataLoader;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB
static uint8_t temp_allocator_pool[1024U * 1024U];       // 1 MB

static bool parse_int(const char* s, int* out) {
  if (!s || !out) {
    return false;
  }
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    return false;
  }
  *out = static_cast<int>(v);
  return true;
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const char* model_path = (argc >= 2) ? argv[1] : "./model.pte";
  int warmup_iters = 5;
  int iters = 50;
  int set_inputs_each_iter = 1; // safer with memory planning
  if (argc >= 3) {
    (void)parse_int(argv[2], &warmup_iters);
  }
  if (argc >= 4) {
    (void)parse_int(argv[3], &iters);
  }
  if (argc >= 5) {
    (void)parse_int(argv[4], &set_inputs_each_iter);
  }

  std::cout << "Model: " << model_path << "\n"
            << "Warmup: " << warmup_iters << ", Iters: " << iters << "\n"
            << "set_inputs_each_iter: " << set_inputs_each_iter << std::endl;

  // Load program from file.
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  if (!loader.ok()) {
    std::cerr << "FileDataLoader::from failed. Error="
              << static_cast<int>(loader.error()) << std::endl;
    return 1;
  }

  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    std::cerr << "Program::load failed. Error="
              << static_cast<int>(program.error()) << std::endl;
    return 2;
  }

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    if (!method_name_result.ok()) {
      std::cerr << "Program has no methods." << std::endl;
      return 3;
    }
    method_name = *method_name_result;
  }
  std::cout << "Method: " << method_name << std::endl;

  Result<MethodMeta> method_meta = program->method_meta(method_name);
  if (!method_meta.ok()) {
    std::cerr << "method_meta failed. Error="
              << static_cast<int>(method_meta.error()) << std::endl;
    return 4;
  }

  // Set up allocators (same idea as executorch/examples/portable/executor_runner).
  MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);
  MemoryAllocator temp_allocator(sizeof(temp_allocator_pool), temp_allocator_pool);

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  const size_t num_planned = method_meta->num_memory_planned_buffers();
  planned_buffers.reserve(num_planned);
  planned_spans.reserve(num_planned);
  for (size_t i = 0; i < num_planned; ++i) {
    const auto sz_res = method_meta->memory_planned_buffer_size(i);
    if (!sz_res.ok()) {
      std::cerr << "memory_planned_buffer_size failed. Error="
                << static_cast<int>(sz_res.error()) << std::endl;
      return 5;
    }
    const size_t buffer_size = static_cast<size_t>(sz_res.get());
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }

  HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(&method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method = program->load_method(method_name, &memory_manager, nullptr);
  if (!method.ok()) {
    std::cerr << "load_method failed. Error=" << static_cast<int>(method.error())
              << std::endl;
    return 6;
  }

  // Create a single input tensor (NCHW float32).
  std::vector<float> input_storage(1 * 3 * 224 * 224, 0.0f);
  TensorPtr input_tensor = executorch::extension::from_blob(
      input_storage.data(), {1, 3, 224, 224});

  // Prepare input EValues.
  std::vector<EValue> inputs;
  inputs.reserve(1);
  inputs.emplace_back(input_tensor); // EValue(TensorPtr) -> EValue(Tensor)

  auto set_inputs = [&]() -> bool {
    const auto err = method->set_inputs(
        executorch::aten::ArrayRef<EValue>(inputs.data(), inputs.size()));
    if (err != Error::Ok) {
      std::cerr << "set_inputs failed. Error=" << static_cast<int>(err) << std::endl;
      return false;
    }
    return true;
  };

  if (!set_inputs()) {
    return 7;
  }

  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    if (set_inputs_each_iter && !set_inputs()) {
      return 7;
    }
    const auto err = method->execute();
    if (err != Error::Ok) {
      std::cerr << "execute (warmup) failed. Error=" << static_cast<int>(err)
                << std::endl;
      return 8;
    }
  }

  // Benchmark
  double sum_ms = 0.0;
  double min_ms = 1e30;
  double max_ms = 0.0;

  for (int i = 0; i < iters; ++i) {
    if (set_inputs_each_iter && !set_inputs()) {
      return 7;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto err = method->execute();
    const auto t1 = std::chrono::steady_clock::now();

    if (err != Error::Ok) {
      std::cerr << "execute failed. Error=" << static_cast<int>(err) << std::endl;
      return 9;
    }

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    sum_ms += ms;
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
  }

  std::cout << "Low-level execute avg/min/max (ms): "
            << (sum_ms / static_cast<double>(iters)) << " / " << min_ms << " / "
            << max_ms << std::endl;

  return 0;
}


