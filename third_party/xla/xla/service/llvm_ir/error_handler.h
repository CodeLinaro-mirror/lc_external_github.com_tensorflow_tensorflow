/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_ERROR_HANDLER_H_
#define XLA_SERVICE_LLVM_IR_ERROR_HANDLER_H_

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"

namespace xla {

// RAII Guard to be used in place of llvm::ScopedFatalErrorHandler
// This class is thread-safe.
// llvm::ScopedFatalErrorHandler uses a static thread handler and consequently
// trying to register a handler on multiple threads causes a race condition and
// a crash.
// This class uses thread local storage to avoid this issue.
// Usage:
//   XlaScopedFatalErrorHandler handler([](absl::string_view reason) {
//     LOG(ERROR) << "Something went wrong:" << reason;
//   });
// The registered lambda itself should be thread-safe as in not accessing any
// shared state between threads without proper synchronization.
class [[nodiscard]] XlaScopedFatalErrorHandler {
 public:
  using Handler = absl::AnyInvocable<void(absl::string_view reason)>;

  // Installs the given handler for the current thread.
  // The previous handler for this thread is kept and restored on destruction.
  explicit XlaScopedFatalErrorHandler(Handler handler = nullptr);
  // Restores the previous handler for this thread.
  ~XlaScopedFatalErrorHandler();

  XlaScopedFatalErrorHandler(const XlaScopedFatalErrorHandler&) = delete;
  XlaScopedFatalErrorHandler(XlaScopedFatalErrorHandler&&) = delete;
  XlaScopedFatalErrorHandler& operator=(const XlaScopedFatalErrorHandler&) =
      delete;
  XlaScopedFatalErrorHandler& operator=(XlaScopedFatalErrorHandler&&) = delete;

 private:
  // RAII guard for the handler.
  Handler handler_;
  // Previous handler for this thread.
  Handler* prev_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_ERROR_HANDLER_H_
