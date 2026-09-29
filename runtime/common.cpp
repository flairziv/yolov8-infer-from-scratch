// common.cpp
#include "common.h"

#include <cstdlib>
#include <fstream>
#include <new>
#include <utility>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace yi {

namespace {
void* aligned_malloc(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, kAlign);                // MSVC 没有 std::aligned_alloc
#else
    return std::aligned_alloc(kAlign, align_up(bytes));   // C11 / C++17 规定：大小必须是对齐值的整数倍
#endif
}

void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

AlignedBuffer::AlignedBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) return;
    ptr_ = aligned_malloc(bytes);
    if (!ptr_) throw std::bad_alloc();
}

AlignedBuffer::~AlignedBuffer() { aligned_free(ptr_); }

// 移动 = 把指针"偷"过来，再把对方置空，这样对方析构时 free(nullptr) 什么也不做
AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        aligned_free(ptr_);
        ptr_ = std::exchange(other.ptr_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

AlignedBuffer read_file(const std::string& path, int64_t expected_bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);   // ate：打开后停在文件尾，tellg() 就是文件大小
    YI_CHECK(f, "打不开文件 " << path);
    const int64_t size = static_cast<int64_t>(f.tellg());
    YI_CHECK(expected_bytes < 0 || size == expected_bytes, path << " 有 " << size << " 字节，预期 " << expected_bytes);
    AlignedBuffer buf(static_cast<size_t>(size));
    f.seekg(0);
    f.read(buf.as<char>(), size);
    YI_CHECK(f, "读取 " << path << " 失败");
    return buf;
}

}  // namespace yi
