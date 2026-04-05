#include "BPlusTree.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <ranges>
#include <stdexcept>

namespace tritondb::storage {
// PageType::BTreeLeaf
//
// prev  : PageId (8)
// next  : PageId (8)
// count : uint16_t (2)
// _pad  : uint16_t (2)
//
inline constexpr uint16_t kLeafHeaderSize = 20;
inline constexpr uint16_t kLeafSlotSize = 12;   // key(8) + offset(2) + len(2)
inline constexpr uint16_t kLeafMinFree = kLeafSlotSize + 1;
// PageType::BTreeInner
//
// count  : uint16_t (2)
// _pad   : uint16_t (2)
// _pad   : uint32_t (4)
// ptr[0] : PageId   (8)
// key[0] : uint64_t (8)
// ...
// ptr[count] : PageId (8)

inline constexpr uint16_t kInnerHeaderSize = 8;
inline constexpr uint16_t kInnerEntrySize = 16;   // key(8) + ptr(8)
inline constexpr uint16_t kInnerMaxKeys =
    static_cast<uint16_t>((kPageDataSize - kInnerHeaderSize - 8) / kInnerEntrySize);
inline constexpr uint16_t kInnerMinKeys = kInnerMaxKeys / 2;

inline constexpr uint32_t kBTreeMagic = 0x54444200U;
inline constexpr uint16_t kMetaSize = 16;

struct LeafSlot {
    BTreeKey key;
    uint16_t valOffset;   // смещение от начална data.
    uint16_t valLen;
};

struct LeafNode {
    PageId prev = kInvalidPageId;
    PageId next = kInvalidPageId;
    std::vector<LeafSlot> slots;
    std::array<std::byte, kPageDataSize> raw {};

    [[nodiscard]] BTreeValue valueAt(std::size_t i) const {
        const auto& s = slots[i];
        BTreeValue v(s.valLen);
        std::memcpy(v.data(), raw.data() + s.valOffset, s.valLen);
        return v;
    }

    [[nodiscard]] uint16_t usedValueBytes() const noexcept {
        return static_cast<uint16_t>(std::ranges::fold_left(slots, uint16_t { 0 }, [](uint16_t acc, const LeafSlot& s) {
            return static_cast<uint16_t>(acc + s.valLen);
        }));
    }

    [[nodiscard]] int freeBytes() const noexcept {
        int used = kLeafHeaderSize + (static_cast<int>(slots.size()) * kLeafSlotSize) + usedValueBytes();
        return static_cast<int>(kPageDataSize) - used;
    }

    [[nodiscard]] bool fits(uint16_t valLen) const noexcept { return freeBytes() >= kLeafSlotSize + valLen; }
};

struct InnerNode {
    std::vector<BTreeKey> keys;
    std::vector<PageId> ptrs;
};

static LeafNode deserializeLeaf(const Page& page, bool isRoot) {
    LeafNode node;
    const auto* d = page.data.data();
    std::size_t offset = isRoot ? kMetaSize : 0;

    std::memcpy(&node.prev, d + offset, sizeof(PageId));
    offset += 8;
    std::memcpy(&node.next, d + offset, sizeof(PageId));
    offset += 8;
    uint16_t count = 0;
    std::memcpy(&count, d + offset, sizeof(uint16_t));
    offset += 2;
    offset += 2;   // _pad

    node.slots.reserve(count);
    for (uint16_t i = 0; i < count; i++) {
        LeafSlot s {};
        std::memcpy(&s.key, d + offset, 8);
        offset += 8;
        std::memcpy(&s.valOffset, d + offset, 2);
        offset += 2;
        std::memcpy(&s.valLen, d + offset, 2);
        offset += 2;
        node.slots.push_back(s);
    }

    node.raw = page.data;
    return node;
}

static void serializeLeaf(const LeafNode& node, Page& page, bool isRoot) {
    auto* d = page.data.data();
    std::size_t offset = isRoot ? kMetaSize : 0;

    auto val_end = static_cast<uint16_t>(kPageDataSize);
    std::vector<LeafSlot> new_slots = node.slots;

    for (auto& s : new_slots) {
        val_end -= s.valLen;
        std::memmove(d + val_end, node.raw.data() + s.valOffset, s.valLen);
        s.valOffset = val_end;
    }

    std::memcpy(d + offset, &node.prev, sizeof(PageId));
    offset += 8;
    std::memcpy(d + offset, &node.next, sizeof(PageId));
    offset += 8;
    auto count = static_cast<uint16_t>(new_slots.size());
    std::memcpy(d + offset, &count, 2);
    offset += 2;
    uint16_t pad = 0;
    std::memcpy(d + offset, &pad, 2);
    offset += 2;

    for (const auto& s : new_slots) {
        std::memcpy(d + offset, &s.key, 8);
        offset += 8;
        std::memcpy(d + offset, &s.valOffset, 2);
        offset += 2;
        std::memcpy(d + offset, &s.valLen, 2);
        offset += 2;
    }

    page.updateChecksum();
}

static InnerNode deserializeInner(const Page& page, bool isRoot) {
    InnerNode node;
    const auto* d = page.data.data();
    std::size_t offset = isRoot ? kMetaSize : 0;

    uint16_t count = 0;
    std::memcpy(&count, d + offset, 2);
    offset += 2 + 6;

    node.ptrs.resize(count + 1);
    node.keys.reserve(count);

    std::memcpy(node.ptrs.data(), d + offset, 8);
    offset += 8;
    for (uint16_t i = 0; i < count; i++) {
        std::memcpy(&node.keys[i], d + offset, 8);
        offset += 8;
        std::memcpy(&node.ptrs[i + 1], d + offset, 8);
        offset += 8;
    }

    return node;
}

static void serializeInner(const InnerNode& node, Page& page, bool isRoot) {
    auto* d = page.data.data();
    std::size_t offset = isRoot ? kMetaSize : 0;

    auto count = static_cast<uint16_t>(node.keys.size());
    std::memcpy(d + offset, &count, 2);
    offset += 2;
    uint16_t p2 = 0;
    std::memcpy(d + offset, &p2, 2);
    offset += 2;
    uint32_t p4 = 0;
    std::memcpy(d + offset, &p4, 4);
    offset += 4;

    std::memcpy(d + offset, &node.ptrs[0], 8);
    offset += 8;
    for (uint16_t i = 0; i < count; i++) {
        std::memcpy(d + offset, &node.keys[i], 8);
        offset += 8;
        std::memcpy(d + offset, &node.ptrs[i + 1], 8);
        offset += 8;
    }
    page.updateChecksum();
}

static void writeMeta(Page& page, uint32_t height, uint64_t size) noexcept {
    auto* d = page.data.data();
    std::memcpy(d, &kBTreeMagic, 4);
    std::memcpy(d + 4, &height, 4);
    std::memcpy(d + 8, &size, 8);
}

static void readMeta(Page& page, uint32_t& height, uint64_t& size) {
    auto* d = page.data.data();
    uint32_t magic = 0;
    std::memcpy(&magic, d, 4);
    std::memcpy(&height, d + 4, 4);
    std::memcpy(&size, d + 8, 8);
    if (magic != kBTreeMagic) {
        throw std::runtime_error(std::format("BPlusTree: bad magic 0x{:08X}", magic));
    }
}

BPlusTree::BPlusTree(IPageManager& pm, PageId rootPageId) : pm_(pm), root_page_id_(rootPageId), size_(0), height_(1) {}

std::unique_ptr<BPlusTree> BPlusTree::create(IPageManager& pm, PageId rootPageId) {
    auto tree = std::unique_ptr<BPlusTree>(new BPlusTree(pm, rootPageId));
    Page page = pm.read(rootPageId);
    page.header.id = rootPageId;
    page.header.type = PageType::BTreeLeaf;
    page.data.fill(std::byte { 0 });

    writeMeta(page, 1, 0);

    LeafNode empty_leaf {};
    serializeLeaf(empty_leaf, page, true);

    pm.write(rootPageId, page);
    return tree;
}

std::unique_ptr<BPlusTree> BPlusTree::open(IPageManager& pm, PageId rootPageId) {
    auto tree = std::unique_ptr<BPlusTree>(new BPlusTree(pm, rootPageId));

    Page page = pm.read(rootPageId);
    readMeta(page, tree->height_, tree->size_);
    return tree;
}

static Page readPage(IPageManager& pm, PageId id, PageType expected) {
    Page p = pm.read(id);
    if (p.header.type != expected) {
        throw std::runtime_error(std::format("BPlusTree: page {} has wrong type", id));
    }
    return p;
}

BPlusTree::SearchPath BPlusTree::findLeaf(BTreeKey key) const {
    SearchPath path;
    PageId cur = root_page_id_;

    for (uint32_t level = 0; level < height_ - 1; level++) {
        path.ancestors.push_back(cur);
        Page page = pm_.read(cur);
        bool is_root = (cur == root_page_id_);
        InnerNode node = deserializeInner(page, is_root);

        auto it = std::ranges::upper_bound(node.keys, key);
        auto idx = static_cast<std::size_t>(std::ranges::distance(node.keys.begin(), it));
        cur = node.ptrs[idx];
    }

    path.leafId = cur;
    return path;
}

std::optional<BTreeValue> BPlusTree::find(BTreeKey key) const {
    SearchPath path = findLeaf(key);
    bool is_root = (path.leafId == root_page_id_);
    Page page = pm_.read(path.leafId);
    LeafNode node = deserializeLeaf(page, is_root);

    auto it = std::ranges::lower_bound(node.slots, key, {}, &LeafSlot::key);

    if (it == node.slots.end() || it->key != key) {
        return std::nullopt;
    }

    return node.valueAt(static_cast<std::size_t>(std::ranges::distance(node.slots.begin(), it)));
}

std::optional<BPlusTree::SplitResult> BPlusTree::insertIntoInner(PageId parentId, BTreeKey separator, PageId newChild) {
    bool is_root = (parentId == root_page_id_);
    Page page = pm_.read(parentId);
    InnerNode node = deserializeInner(page, is_root);

    auto it = std::ranges::upper_bound(node.keys, separator);
    std::size_t pos = static_cast<std::size_t>(std::ranges::distance(node.keys.begin(), it));
    node.keys.insert(it, separator);
    node.ptrs.insert(node.ptrs.begin() + static_cast<std::ptrdiff_t>(pos) + 1, newChild);

    if (node.keys.size() <= kInnerMaxKeys) {
        serializeInner(node, page, is_root);
        pm_.write(parentId, page);
        return std::nullopt;
    }

    return splitInnerImpl(parentId, &node);
}

std::optional<BPlusTree::SplitResult>
BPlusTree::insertIntoLeaf(PageId leafId, BTreeKey key, std::span<const std::byte> value) {
    bool is_root = (leafId == root_page_id_);
    Page page = pm_.read(leafId);
    LeafNode node = deserializeLeaf(page, is_root);

    auto it = std::ranges::lower_bound(node.slots, key, {}, &LeafSlot::key);

    if (it != node.slots.end() && it->key == key) {
        std::size_t idx = static_cast<std::size_t>(std::ranges::distance(node.slots.begin(), it));
        auto& s = node.slots[idx];

        int delta = static_cast<int>(value.size()) - static_cast<int>(s.valLen);
        if (node.freeBytes() - delta >= 0) {
            BTreeValue new_val(value.begin(), value.end());
            s.valLen = static_cast<uint16_t>(value.size());
            std::memcpy(node.raw.data() + s.valOffset, value.data(), value.size());
            serializeLeaf(node, page, is_root);
            pm_.write(leafId, page);
            return std::nullopt;
        }
        node.slots.erase(it);
        size_--;
        it = std::ranges::lower_bound(node.slots, key, {}, &LeafSlot::key);
    }
    size_++;

    auto new_val_offset = static_cast<uint16_t>(kPageDataSize - 1);
    LeafSlot new_slot { .key = key, .valOffset = new_val_offset, .valLen = static_cast<uint16_t>(value.size()) };

    std::size_t raw_end = kPageDataSize - node.usedValueBytes() - value.size();
    std::ranges::copy(value, node.raw.data() + raw_end);
    new_slot.valOffset = static_cast<uint16_t>(raw_end);

    node.slots.insert(it, new_slot);
    if (node.fits(0) || node.freeBytes() >= 0) {
        serializeLeaf(node, page, is_root);
        pm_.write(leafId, page);
        return std::nullopt;
    }

    return splitLeafImpl(leafId, &node);
}

void BPlusTree::insert(BTreeKey key, std::span<const std::byte> value) {
    if (value.size() > (kPageDataSize - kLeafHeaderSize) / 2) {
        throw std::runtime_error("BPlusTree::insert: value too large");
    }

    SearchPath path = findLeaf(key);

    auto split_res = insertIntoLeaf(path.leafId, key, value);

    if (!split_res) {
        updateRootMeta();
        return;
    }

    BTreeKey sep = split_res->separator;
    PageId right_id = split_res->newPageId;

    for (PageId parent_id : path.ancestors | std::views::reverse) {
        auto inner_split = insertIntoInner(parent_id, sep, right_id);
        if (!inner_split) {
            updateRootMeta();
            return;
        }
        sep = inner_split->separator;
        right_id = inner_split->newPageId;
    }

    growRoot(sep, right_id);
}
}   // namespace tritondb::storage
