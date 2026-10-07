// thread_pool.h —— 固定线程数的线程池：把 [0, n) 的工作单元动态分给各线程执行。
// 只服务于单个算子内部的数据并行：并行区间不嵌套，工作单元之间必须互不影响。
// 工作单元彼此独立 ⇒ 谁先谁后不影响结果，输出与单线程逐位一致。
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace yi {

class ThreadPool {
public:
    explicit ThreadPool(int threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return threads_; }

    // 把 [0, count) 的单元并发执行；fn(i) 会被任意线程调用，i 互不相同。
    // 任一单元抛异常时其余单元尽快停下，异常在调用线程重新抛出。
    void parallel_for(int64_t count, const std::function<void(int64_t)>& fn);

    // 当前线程在池里的槽位：调用线程固定 0，工作线程领取 1..threads-1。
    // 算子用它给每个线程分配私有暂存区（例如 im2col 面板）。
    static int worker_slot();

private:
    void worker_loop();
    void run_units();

    const int threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_, done_cv_;
    const std::function<void(int64_t)>* fn_ = nullptr;   // 本轮任务；parallel_for 返回前保持有效
    int64_t count_ = 0;
    std::atomic<int64_t> next_{0};                       // 下一个待领取的单元号
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    int pending_ = 0;                                    // 还没结束本轮任务的工作线程数
    uint64_t generation_ = 0;
    bool stop_ = false;
};

}  // namespace yi
