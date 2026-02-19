/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_CUSTOM_CALL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_CUSTOM_CALL_H_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "xla/ffi/type_registry.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace mosaic::gpu {

using MosaicInitFunc = void(void**, void**);
using MosaicHostFunc = void(void*, void*, void**);
using KernelHash = std::array<uint64_t, 4>;

struct CompiledKernel {
  CompiledKernel(std::unique_ptr<llvm::orc::LLJIT> lljit,
                 std::string object_file, MosaicHostFunc* host_launch,
                 MosaicInitFunc* init, bool is_comm_used,
                 std::string host_func_name, std::string init_func_name)
      : lljit(std::move(lljit)),
        object_file(std::move(object_file)),
        host_launch(host_launch),
        init(init),
        is_comm_used(is_comm_used),
        host_func_name(std::move(host_func_name)),
        init_func_name(std::move(init_func_name)) {}

  // CompiledKernel is neither copyable nor movable. We use CompiledKernel* as a
  // key in a cache, so we require pointer stability.
  CompiledKernel(const CompiledKernel&) = delete;
  CompiledKernel(CompiledKernel&& other) = delete;

  std::unique_ptr<llvm::orc::LLJIT> lljit;
  std::string object_file;
  MosaicHostFunc* host_launch = nullptr;
  MosaicInitFunc* init = nullptr;
  bool is_comm_used = false;
  std::string host_func_name;
  std::string init_func_name;
};

// TODO(b/464203195): Require a compute capability to be passed in.
absl::StatusOr<std::unique_ptr<CompiledKernel>> Compile(
    llvm::StringRef module_str,
    stream_executor::CudaComputeCapability cc);

struct CustomCallResources {
  CompiledKernel* kernel;
  KernelHash hash;

  static absl::StatusOr<std::string> Serialize(
      const CustomCallResources& resources);
  static absl::StatusOr<std::unique_ptr<CustomCallResources>> Deserialize(
      absl::string_view data);
};

}  // namespace mosaic::gpu

namespace xla::ffi {
template <>
struct TypeRegistry::SerDes<mosaic::gpu::CustomCallResources> {
  static constexpr bool value = true;

  static absl::StatusOr<std::string> Serialize(
      const mosaic::gpu::CustomCallResources& resources) {
    return mosaic::gpu::CustomCallResources::Serialize(resources);
  }

  static absl::StatusOr<std::unique_ptr<mosaic::gpu::CustomCallResources>>
  Deserialize(absl::string_view data) {
    return mosaic::gpu::CustomCallResources::Deserialize(data);
  }
};
}  // namespace xla::ffi

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_CUSTOM_CALL_H_
