#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "Page.h"
#include "PageManager.h"

namespace tritondb::storage {

using BTreeKey = uint64_t;
using BTreeValue = std::vector<std::byte>;

struct BTreeEntry {
    BTreeKey key;
    BTreeValue value;
};

class IBPlusTree {
public:
    virtual ~IBPlusTree() = default;
    virtual void insert(BTreeKey key, std::span<const std::byte> value) = 0;
    [[nodiscard]] virtual std::optional<BTreeValue> find(BTreeKey key) const = 0;
    virtual void remove(BTreeKey key) = 0;
    virtual void rangeScan(BTreeKey from, BTreeKey to, std::function<bool(const BTreeEntry&)> callback) const = 0;
    virtual void scan(std::function<bool(const BTreeEntry&)> callback) const = 0;
    [[nodiscard]] virtual uint64_t size() const noexcept = 0;
    [[nodiscard]] virtual uint32_t height() const noexcept = 0;
};

class BPlusTree final : public IBPlusTree {
public:
    [[nodiscard]] static std::unique_ptr<BPlusTree> create(IPageManager& pm, PageId rootPageId);

    [[nodiscard]] static std::unique_ptr<BPlusTree> open(IPageManager& pm, PageId rootPageId);

    ~BPlusTree() override = default;

    void insert(BTreeKey key, std::span<const std::byte> value) override;
    [[nodiscard]] std::optional<BTreeValue> find(BTreeKey key) const override;
    void remove(BTreeKey key) override;
    void rangeScan(BTreeKey from, BTreeKey to, std::function<bool(const BTreeEntry&)> callback) const override;
    void scan(std::function<bool(const BTreeEntry&)> callback) const override;
    [[nodiscard]] uint64_t size() const noexcept override;
    [[nodiscard]] uint32_t height() const noexcept override;

private:
    BPlusTree(IPageManager& pm, PageId rootPageId);

    IPageManager& pm_;
    PageId root_page_id_;
    uint64_t size_;
    uint32_t height_;

    struct SearchPath {
        std::vector<PageId> ancestors;
        PageId leafId;
    };
    [[nodiscard]] SearchPath findLeaf(BTreeKey key) const;

    struct SplitResult {
        BTreeKey separator;
        PageId newPageId;
    };
    SplitResult splitLeafImpl(PageId leafId, void* nodePtr);
    SplitResult splitInnerImpl(PageId innerId, void* nodePtr);

    std::optional<SplitResult> insertIntoLeaf(PageId leafId, BTreeKey key, std::span<const std::byte> value);
    std::optional<SplitResult> insertIntoInner(PageId parentId, BTreeKey separator, PageId newChild);

    void growRoot(BTreeKey separator, PageId rightChild);

    void updateRootMeta();

    void rebalanceLeaf(const SearchPath& path);
    void rebalanceInner(const SearchPath& path);
};

inline constexpr std::size_t kLeafMaxEntries = 200;
inline constexpr std::size_t kInnerMaxEntries = 340;

}   // namespace tritondb::storage