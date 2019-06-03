#include "tensorflow/compiler/xla/xla_client/multi_wait.h"

#include <chrono>
#include <exception>

#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace util {

void MultiWait::Done(Status status) {
  bool notify = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_count_ += 1;
    notify = completed_count_ >= count_;
    if (!status.ok() && status_.ok()) {
      status_ = status;
    }
  }
  if (notify) {
    cv_.notify_all();
  }
}

Status MultiWait::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return completed_count_ >= count_; });
  return status_;
}

Status MultiWait::Wait(double wait_seconds) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!cv_.wait_for(lock, std::chrono::duration<double>(wait_seconds),
                    [this] { return completed_count_ >= count_; })) {
    return tensorflow::errors::DeadlineExceeded("Timeout");
  }
  return status_;
}

void MultiWait::Reset(size_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  count_ = count;
  completed_count_ = 0;
  status_ = Status::OK();
}

std::function<void()> MultiWait::Completer(std::function<void()> func) {
  auto completer = [this, func = std::move(func)]() {
    try {
      func();
      Done();
    } catch (const std::exception& ex) {
      Done(tensorflow::errors::Internal(ex.what()));
    }
  };
  return completer;
}

}  // namespace util
}  // namespace xla
