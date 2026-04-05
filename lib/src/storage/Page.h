#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tritondb::storage {

inline constexpr std::size_t kPageSize = 4096;
inline constexpr std::size_t kPageHeaderSize = 16;
inline constexpr std::size_t kPageDataSize = kPageSize - kPageHeaderSize;

using PageId = std::uint64_t;
inline constexpr PageId kInvalidPageId = UINT64_MAX;

enum class PageType : uint8_t {
    Free = 0,
    TreeInner = 1,
    TreeLeaf = 2,
    WalSegment = 3,
    FreeList = 4,
};

struct PageHeader {
    PageId id;
    PageType type;
    uint8_t reserved;
    uint16_t checksum;
    uint32_t lsn;
};
static_assert(sizeof(PageHeader) == kPageHeaderSize, "Page header size mismatch");

struct Page {
    PageHeader header;
    std::array<std::byte, kPageDataSize> data;

    void clear(PageType type) noexcept;

    void updateChecksum() noexcept;

    bool verifyChecksum() noexcept;
};
static_assert(sizeof(Page) == kPageSize, "Page size mismatch");

}   // namespace tritondb::storage
