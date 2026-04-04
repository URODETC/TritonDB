#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "Page.h"

namespace tritondb::storage {

class IPageManager {
public:
    virtual ~IPageManager() = default;

    virtual PageId alloc(PageType type) = 0;

    virtual Page read(PageId id) const = 0;

    virtual void write(PageId id, Page page) = 0;

    virtual void free(PageId id) = 0;

    virtual void sync() = 0;

    virtual uint64_t pageCount() const noexcept = 0;
};

class PageManager final : public IPageManager {
public:
    static std::unique_ptr<PageManager> open(const std::filesystem::path& path);

    ~PageManager() override;

    PageId alloc(PageType type) override;
    Page read(PageId id) const override;
    void write(PageId id, Page page) override;
    void free(PageId id) override;
    void sync() override;
    uint64_t pageCount() const noexcept override;

private:
    explicit PageManager(int fd, uint64_t pageCount);

    int fd_;
    uint64_t pageCount_;

    std::vector<PageId> freeList_;

    off_t offsetOf(PageId id) const noexcept;
    void loadFreeList();
    void persistFreeList();
};

}   // namespace tritondb::storage