// common.h —— 全工程共用的小工具：检查宏、64 字节对齐的内存块、读二进制文件
#pragma once
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

// 永远生效的检查：不像 assert 会被 Release 的 -DNDEBUG 关掉。失败时抛异常，消息里带文件名和行号。
// msg 部分可以用 << 拼接，例如 YI_CHECK(n > 0, "张量 " << name << " 是空的")
#define YI_CHECK(cond, msg)                                                           \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::ostringstream yi_oss_;                                               \
            yi_oss_ << __FILE__ << ":" << __LINE__ << " 检查失败 [" #cond "] " << msg; \
            throw std::runtime_error(yi_oss_.str());                                  \
        }                                                                             \
    } while (0)

namespace yi {

// 64 字节 = 一条缓存行；同时满足 SSE（16）、AVX（32）、AVX-512（64）对齐读写的要求
constexpr size_t kAlign = 64;

inline size_t align_up(size_t n, size_t a = kAlign) { return (n + a - 1) / a * a; }

// 一块 64 字节对齐的内存，RAII 管理：析构时自动释放。
// 只能移动、不能复制 —— 复制会让两个对象指向同一块内存，析构时 free 两次。
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t bytes);
    ~AlignedBuffer();
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    template <class T>
    T* as() const { return static_cast<T*>(ptr_); }
    size_t bytes() const { return bytes_; }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

// 把整个文件读进一块对齐内存；expected_bytes >= 0 时核对文件大小
AlignedBuffer read_file(const std::string& path, int64_t expected_bytes = -1);

}  // namespace yi
