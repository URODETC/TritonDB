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
    node.keys.resize(count);

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
    if (node.fits(value.size())) {
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

BPlusTree::SplitResult BPlusTree::splitLeafImpl(PageId leafId, void* nodePtr) {
    LeafNode& left = *static_cast<LeafNode*>(nodePtr);
    bool is_root = (leafId == root_page_id_);

    std::size_t mid = left.slots.size() / 2;

    LeafNode right;
    auto mid_it = left.slots.begin() + static_cast<std::ptrdiff_t>(mid);
    right.slots.assign(mid_it, left.slots.end());
    left.slots.erase(mid_it, left.slots.end());

    right.raw = left.raw;
    PageId right_id = pm_.alloc(PageType::BTreeLeaf);

    right.prev = leafId;
    right.next = left.next;
    left.next = right_id;

    if (right.next != kInvalidPageId) {
        Page neighbor_page = pm_.read(right.next);
        bool neighbor_is_root = (right.next == root_page_id_);
        LeafNode neighbor = deserializeLeaf(neighbor_page, neighbor_is_root);
        neighbor.prev = right_id;
        serializeLeaf(neighbor, neighbor_page, neighbor_is_root);
        pm_.write(right.next, neighbor_page);
    }

    Page left_page = pm_.read(leafId);
    Page right_page = pm_.read(right_id);
    right_page.header.id = right_id;
    right_page.header.type = PageType::BTreeLeaf;
    serializeLeaf(left, left_page, is_root);
    serializeLeaf(right, right_page, false);

    pm_.write(leafId, left_page);
    pm_.write(right_id, right_page);

    BTreeKey separator = right.slots.front().key;
    return { .separator = separator, .newPageId = right_id };
}

BPlusTree::SplitResult BPlusTree::splitInnerImpl(PageId innerId, void* nodePtr) {
    InnerNode& left = *static_cast<InnerNode*>(nodePtr);
    bool is_root = (innerId == root_page_id_);

    std::size_t mid = left.keys.size() / 2;
    BTreeKey separator = left.keys[mid];

    InnerNode right;
    auto key_mid_it = left.keys.begin() + static_cast<std::ptrdiff_t>(mid) + 1;
    auto ptr_mid_it = left.ptrs.begin() + static_cast<std::ptrdiff_t>(mid) + 1;
    right.keys.assign(key_mid_it, left.keys.end());
    right.ptrs.assign(ptr_mid_it, left.ptrs.end());

    left.keys.erase(left.keys.begin() + static_cast<std::ptrdiff_t>(mid), left.keys.end());
    left.ptrs.erase(ptr_mid_it, left.ptrs.end());

    PageId right_id = pm_.alloc(PageType::BTreeInner);

    Page left_page = pm_.read(innerId);
    Page right_page = pm_.read(right_id);
    right_page.header.id = right_id;
    right_page.header.type = PageType::BTreeInner;
    serializeInner(left, left_page, is_root);
    serializeInner(right, right_page, false);

    pm_.write(innerId, left_page);
    pm_.write(right_id, right_page);

    return { .separator = separator, .newPageId = right_id };
}

void BPlusTree::growRoot(BTreeKey separator, PageId rightChild) {
    PageId new_root_id = pm_.alloc(PageType::BTreeInner);

    PageId left_child = pm_.alloc(height_ == 1 ? PageType::BTreeLeaf : PageType::BTreeInner);
    Page old_root = pm_.read(root_page_id_);
    Page left_page = pm_.read(left_child);
    left_page.data = old_root.data;
    left_page.header = old_root.header;
    left_page.header.id = left_child;
    left_page.header.type = (height_ == 1) ? PageType::BTreeLeaf : PageType::BTreeInner;

    if (height_ == 1) {
        LeafNode leaf_node = deserializeLeaf(old_root, true);
        serializeLeaf(leaf_node, left_page, false);
    } else {
        InnerNode inner_node = deserializeInner(old_root, true);
        serializeInner(inner_node, left_page, false);
    }
    pm_.write(left_child, left_page);

    if (height_ == 1) {
        Page right_page = pm_.read(rightChild);
        LeafNode right_leaf = deserializeLeaf(right_page, false);
        right_leaf.prev = left_child;
        serializeLeaf(right_leaf, right_page, false);
        pm_.write(rightChild, right_page);
    }

    InnerNode new_root;
    new_root.keys = { separator };
    new_root.ptrs = { left_child, rightChild };

    Page new_root_page = pm_.read(new_root_id);
    new_root_page.header.id = new_root_id;
    new_root_page.header.type = PageType::BTreeInner;

    new_root_page.data.fill(std::byte { 0 });
    serializeInner(new_root, new_root_page, true);

    pm_.write(root_page_id_, new_root_page);
    pm_.free(new_root_id);

    height_++;

    updateRootMeta();
}

void BPlusTree::updateRootMeta() {
    Page page = pm_.read(root_page_id_);
    writeMeta(page, height_, size_);
    page.updateChecksum();
    pm_.write(root_page_id_, page);
}

void BPlusTree::remove(BTreeKey key) {
    SearchPath path = findLeaf(key);
    bool is_root = (path.leafId == root_page_id_);
    Page page = pm_.read(path.leafId);
    LeafNode node = deserializeLeaf(page, is_root);
    auto it = std::ranges::lower_bound(node.slots, key, {}, &LeafSlot::key);

    if (it == node.slots.end() || it->key != key) {
        return;
    }

    node.slots.erase(it);
    size_--;
    serializeLeaf(node, page, is_root);
    pm_.write(path.leafId, page);

    if (!path.ancestors.empty() && node.slots.empty()) {
        rebalanceLeaf(path);
    }
    updateRootMeta();
}
// TODO: текущая реализация хуйня сделать нормальный merge/redistribution
void BPlusTree::rebalanceLeaf(const SearchPath& path) {
    PageId leaf_id = path.leafId;
    PageId parent_id = path.ancestors.back();
    bool parent_is_root = (parent_id == root_page_id_);

    Page parent_page = pm_.read(parent_id);
    InnerNode parent = deserializeInner(parent_page, parent_is_root);

    auto pit = std::ranges::find(parent.ptrs, leaf_id);
    if (pit == parent.ptrs.end())
        return;

    std::size_t p_idx = static_cast<std::size_t>(std::ranges::distance(parent.ptrs.begin(), pit));
    Page leaf_page = pm_.read(leaf_id);
    bool leaf_is_root = (leaf_id == root_page_id_);
    LeafNode leaf = deserializeLeaf(leaf_page, leaf_is_root);

    if (leaf.prev != kInvalidPageId) {
        Page prev_page = pm_.read(leaf.prev);
        bool prev_is_root = (leaf.prev == root_page_id_);
        LeafNode prev_node = deserializeLeaf(prev_page, prev_is_root);
        prev_node.next = leaf.next;
        serializeLeaf(prev_node, prev_page, prev_is_root);
        pm_.write(leaf.prev, prev_page);
    }
    if (leaf.next != kInvalidPageId) {
        Page next_page = pm_.read(leaf.next);
        bool next_is_root = (leaf.next == root_page_id_);
        LeafNode next_node = deserializeLeaf(next_page, next_is_root);
        next_node.prev = leaf.prev;
        serializeLeaf(next_node, next_page, next_is_root);
        pm_.write(leaf.next, next_page);
    }
    if (p_idx == 0) {
        parent.keys.erase(parent.keys.begin());
    } else {
        parent.keys.erase(parent.keys.begin() + static_cast<std::ptrdiff_t>(p_idx) - 1);
    }
    parent.ptrs.erase(pit);
    pm_.free(leaf_id);

    if (parent.ptrs.size() - 1 && parent_is_root && height_ > 1) {
        PageId only_child = parent.ptrs[0];
        Page child_page = pm_.read(only_child);
        bool child_is_leaf = (child_page.header.type == PageType::BTreeLeaf);

        if (child_is_leaf) {
            LeafNode child_leaf = deserializeLeaf(child_page, false);
            Page root_page = pm_.read(root_page_id_);
            root_page.header.type = PageType::BTreeLeaf;
            serializeLeaf(child_leaf, root_page, true);
            pm_.write(root_page_id_, root_page);
        } else {
            InnerNode child_inner = deserializeInner(child_page, false);
            Page root_page = pm_.read(root_page_id_);
            root_page.header.type = PageType::BTreeInner;
            serializeInner(child_inner, root_page, true);
            pm_.write(root_page_id_, root_page);
        }
        pm_.free(only_child);
        height_--;
        return;
    }

    serializeInner(parent, parent_page, parent_is_root);
    pm_.write(parent_id, parent_page);

    if (parent.keys.empty() && !parent_is_root && path.ancestors.size() > 1) {
        SearchPath parent_path;
        parent_path.ancestors.assign(path.ancestors.begin(), path.ancestors.end() - 1);
        parent_path.leafId = parent_id;
        rebalanceInner(parent_path);
    }
}

void BPlusTree::rebalanceInner(const SearchPath& path) {
    if (path.ancestors.empty())
        return;

    PageId node_id = path.leafId;
    PageId parent_id = path.ancestors.back();
    bool parent_is_root = (parent_id == root_page_id_);

    Page parent_page = pm_.read(parent_id);
    InnerNode parent = deserializeInner(parent_page, parent_is_root);

    auto pit = std::ranges::find(parent.ptrs, node_id);
    if (pit == parent.ptrs.end())
        return;
    std::size_t p_idx = static_cast<std::size_t>(std::ranges::distance(parent.ptrs.begin(), pit));

    if (p_idx == 0) {
        parent.keys.erase(parent.keys.begin());
    } else {
        parent.keys.erase(parent.keys.begin() + static_cast<ptrdiff_t>(p_idx) - 1);
    }
    parent.ptrs.erase(pit);
    pm_.free(node_id);

    if (parent.ptrs.size() - 1 && parent_is_root && height_ > 1) {
        PageId only_child = parent.ptrs[0];
        Page child_page = pm_.read(only_child);
        bool child_is_leaf = (child_page.header.type == PageType::BTreeLeaf);

        if (child_is_leaf) {
            LeafNode child_leaf = deserializeLeaf(child_page, false);
            Page root_page = pm_.read(root_page_id_);
            root_page.header.type = PageType::BTreeLeaf;
            serializeLeaf(child_leaf, root_page, true);
            pm_.write(root_page_id_, root_page);
        } else {
            InnerNode child_inner = deserializeInner(child_page, false);
            Page root_page = pm_.read(root_page_id_);
            root_page.header.type = PageType::BTreeInner;
            serializeInner(child_inner, root_page, true);
            pm_.write(root_page_id_, root_page);
        }
        pm_.free(only_child);
        height_--;
        return;
    }

    serializeInner(parent, parent_page, parent_is_root);
    pm_.write(parent_id, parent_page);
}

void BPlusTree::scan(std::function<bool(const BTreeEntry&)> callback) const {
    rangeScan(0, UINT64_MAX, std::move(callback));
}

void BPlusTree::rangeScan(BTreeKey from, BTreeKey to, std::function<bool(const BTreeEntry&)> callback) const {
    if (from >= to)
        return;
    SearchPath path = findLeaf(from);
    PageId cur = path.leafId;

    while (cur != kInvalidPageId) {
        bool is_root = (cur == root_page_id_);
        Page page = pm_.read(cur);
        LeafNode node = deserializeLeaf(page, is_root);

        std::size_t i = 0;
        for (auto s : node.slots) {
            if (s.key < from) {
                i++;
                continue;
            }
            if (s.key >= to)
                return;

            BTreeEntry entry { .key = s.key, .value = node.valueAt(i) };
            if (!callback(entry))
                return;
            i++;
        }
        cur = node.next;
    }
}

uint64_t BPlusTree::size() const noexcept { return size_; }
uint32_t BPlusTree::height() const noexcept { return height_; }
}   // namespace tritondb::storage
