#pragma once

#include <cstddef>
#include <cstdint>

#include "Page.h"

namespace tritondb::storage {

enum class NodeType : uint8_t { kInternal = 0, kLeaf = 1 };

struct TreePageHeader {
    NodeType type { NodeType::kLeaf };
    std::uint8_t num_keys { 0 };
    PageId parent_id { kInvalidPageId };
    PageId next_page_id { kInvalidPageId };
    PageId prev_page_id { kInvalidPageId };
};

}   // namespace tritondb::storage