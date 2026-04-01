#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tritondb::storage {

using PageId = std::uint64_t;
inline constexpr PageId kInvalidPageId = static_cast<PageId>(-1);
inline constexpr std::size_t kPageSize = 4096;

struct Page {
    PageId id;
    std::array<std::byte, kPageSize> data {};
};

}   // namespace tritondb::storage
