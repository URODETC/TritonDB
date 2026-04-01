#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "PageManager.h"
#include "TreeNode.h"

namespace tritondb::storage {
class Tree {
public:
    explicit Tree(PageManager& pm, PageId root_id = kInvalidPageId);

    bool insert(uint64_t key, std::span<const std::byte> value) const;

    std::optional<std::vector<std::byte>> find(uint64_t key) const;

private:
    PageManager& pm_;
    PageId root_id_;

    void initialize_root();
};

}   // namespace tritondb::storage