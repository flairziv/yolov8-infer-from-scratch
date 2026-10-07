// thread_pool.cpp —— 简单线程池：每轮任务一个“代”（generation），工作线程睡在条件变量上。
#include "thread_pool.h"

#include "common.h"

namespace yi {
namespace {
// 每个线程一个槽位：调用线程固定 0，工作线程启动时领取 1..threads-1。
thread_local int tls_slot = 0;
}  // namespace

int ThreadPool::worker_slot() { return tls_slot; }

ThreadPool::ThreadPool(int threads) : threads_(threads) {
    YI_CHECK(threads >= 1, "线程数至少是 1");
    for (int i = 1; i < threads_; ++i)
        workers_.emplace_back([this, i] {
            tls_slot = i;
            worker_loop();
        });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::run_units() {
    // 动态领取：先做完的线程接着领下一块，各层工作量不齐时也不会空等。
    for (int64_t i = next_.fetch_add(1, std::memory_order_relaxed); i < count_;
         i = next_.fetch_add(1, std::memory_order_relaxed)) {
        if (failed_.load(std::memory_order_relaxed)) return;
        try {
            (*fn_)(i);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
            }
            failed_.store(true, std::memory_order_relaxed);
            return;
        }
    }
}

void ThreadPool::worker_loop() {
    uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        lock.unlock();
        run_units();
        lock.lock();
        if (--pending_ == 0) done_cv_.notify_all();
    }
}

void ThreadPool::parallel_for(int64_t count, const std::function<void(int64_t)>& fn) {
    if (count <= 0) return;
    if (threads_ == 1 || count == 1) {   // 单线程池或只有一个单元：就地执行，不唤醒任何线程
        for (int64_t i = 0; i < count; ++i) fn(i);
        return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    fn_ = &fn;
    count_ = count;
    next_.store(0, std::memory_order_relaxed);
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    pending_ = threads_ - 1;
    ++generation_;
    work_cv_.notify_all();
    lock.unlock();

    run_units();   // 调用线程也干活，不干等的线程不浪费核

    lock.lock();
    done_cv_.wait(lock, [&] { return pending_ == 0; });
    fn_ = nullptr;
    if (error_) std::rethrow_exception(error_);
}

}  // namespace yi
