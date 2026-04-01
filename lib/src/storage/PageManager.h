#pragma once

#include <filesystem>
#include <fstream>

#include "Page.h"

namespace tritondb::storage {

class PageManager {
public:
    explicit PageManager(std::filesystem::path file_path);
    ~PageManager() = default;

    PageManager(const PageManager&) = delete;
    PageManager& operator=(const PageManager&) = delete;

    PageManager(PageManager&&) = delete;

    [[nodiscard]] PageId alloc();
    void write(const Page& page);
    [[nodiscard]] Page read(PageId id) const;

private:
    std::filesystem::path file_path_;
    mutable std::fstream file_;

    std::uint64_t page_count() const;
    void ensure_open() const;
};

}   // namespace tritondb::storage