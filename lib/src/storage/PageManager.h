#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "Page.h"

namespace tritondb::storage {

inline constexpr PageId kFreeListHeadPageId = 0;
inline constexpr uint32_t kFreeListPageCapacity = static_cast<uint32_t>((kPageDataSize - sizeof(PageId) - sizeof(uint32_t)) / sizeof(PageId));

class IPageManager {
public:
    virtual ~IPageManager() = default;

    [[nodiscard]] virtual PageId alloc(PageType type) = 0;
    [[nodiscard]] virtual Page read(PageId id) const = 0;
    virtual void write(PageId id, Page page) = 0;
    virtual void free(PageId id) = 0;
    virtual void sync() = 0;
    [[nodiscard]] virtual uint64_t pageCount() const noexcept = 0;
    [[nodiscard]] virtual uint64_t freeCount() const noexcept = 0;
};

class PageManager final : public IPageManager {
public:
    [[nodiscard]] static std::unique_ptr<PageManager> open(const std::filesystem::path& path);

    ~PageManager() override;

    [[nodiscard]]PageId alloc(PageType type) override;
    [[nodiscard]] Page read(PageId id) const override;
    void write(PageId id, Page page) override;
    void free(PageId id) override;
    void sync() override;
    [[nodiscard]] uint64_t pageCount() const noexcept override;
    [[nodiscard]] uint64_t freeCount() const noexcept override;

private:
    explicit PageManager(int fd, uint64_t pageCount);

    int fd_;
    uint64_t pageCount_;

    struct FreeListPage {
        PageId nextPageId = kInvalidPageId;
        std::vector<PageId> ids;
    };

    FreeListPage freeListHead_;

    static void serializeFreeListPage(const FreeListPage& flPage, Page& page) noexcept;
    static FreeListPage deserializeFreeListPage(const Page& page);

    [[nodiscard]] FreeListPage readFreeListPage(PageId flPageId) const;

    void writeFreeListPage(PageId flPageId, const FreeListPage& flPage);
    void flushHead();
    [[nodiscard]] PageId popFree();
    void pushFree(PageId id);
    [[nodiscard]] PageId extendFile();
    [[nodiscard]] static off_t offsetOf(PageId id) noexcept;

    void loadHead();
};

}   // namespace tritondb::storage