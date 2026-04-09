#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <ranges>

#include "storage/BPlusTree.h"
#include "storage/PageManager.h"

using namespace tritondb::storage;

class BPlusTreeTest : public ::testing::Test {
protected:
    std::filesystem::path dbPath;
    std::unique_ptr<PageManager> pm;
    std::unique_ptr<IBPlusTree> tree;
    PageId rootPageId;

    void SetUp() override {
        dbPath = std::filesystem::temp_directory_path() / "tritondb_btree_test.db";
        std::filesystem::remove(dbPath);
        pm = PageManager::open(dbPath);
        rootPageId = pm->alloc(PageType::BTreeLeaf);
        tree = BPlusTree::create(*pm, rootPageId);
    }

    void TearDown() override {
        tree.reset();
        pm.reset();
        std::filesystem::remove(dbPath);
    }

    void insertUint(BTreeKey key, uint64_t val) {
        auto bytes = std::as_bytes(std::span { &val, 1 });
        tree->insert(key, bytes);
    }

    uint64_t readUint(const BTreeValue& val) {
        uint64_t result = 0;
        std::memcpy(&result, val.data(), sizeof(result));
        return result;
    }

    void insertRange(int n, uint64_t multiplier = 1) {
        for (int i = 0; i < n; ++i) insertUint(static_cast<BTreeKey>(i), static_cast<uint64_t>(i) * multiplier);
    }

    std::vector<BTreeKey> collectKeys() {
        std::vector<BTreeKey> keys;
        tree->scan([&](const BTreeEntry& e) {
            keys.push_back(e.key);
            return true;
        });
        return keys;
    }
};

TEST_F(BPlusTreeTest, NewTreeIsEmpty) {
    EXPECT_EQ(tree->size(), 0u);
    EXPECT_EQ(tree->height(), 1u);
}

TEST_F(BPlusTreeTest, InsertOneAndFind) {
    insertUint(42, 1000);
    auto result = tree->find(42);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(readUint(*result), 1000u);
}

TEST_F(BPlusTreeTest, FindMissingKeyReturnsNullopt) { EXPECT_FALSE(tree->find(999).has_value()); }

TEST_F(BPlusTreeTest, InsertUpdatesExistingKey) {
    insertUint(1, 100);
    insertUint(1, 200);
    auto result = tree->find(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(readUint(*result), 200u);
}

TEST_F(BPlusTreeTest, InsertIncrementsSize) {
    EXPECT_EQ(tree->size(), 0u);
    insertUint(1, 10);
    EXPECT_EQ(tree->size(), 1u);
    insertUint(2, 20);
    EXPECT_EQ(tree->size(), 2u);
    insertUint(1, 99);
    EXPECT_EQ(tree->size(), 2u);
}

TEST_F(BPlusTreeTest, RemoveExistingKey) {
    insertUint(5, 50);
    tree->remove(5);
    EXPECT_FALSE(tree->find(5).has_value());
    EXPECT_EQ(tree->size(), 0u);
}

TEST_F(BPlusTreeTest, RemoveMissingKeyDoesNotThrow) { EXPECT_NO_THROW(tree->remove(999)); }

TEST_F(BPlusTreeTest, RemoveDecrementsSize) {
    insertRange(10);
    EXPECT_EQ(tree->size(), 10u);
    tree->remove(3);
    EXPECT_EQ(tree->size(), 9u);
    tree->remove(3);
    EXPECT_EQ(tree->size(), 9u);
}

TEST_F(BPlusTreeTest, RemoveOnlyKeyLeavesEmptyTree) {
    insertUint(7, 77);
    tree->remove(7);
    EXPECT_EQ(tree->size(), 0u);
    EXPECT_EQ(tree->height(), 1u);
    EXPECT_FALSE(tree->find(7).has_value());
}

TEST_F(BPlusTreeTest, InsertManyKeysAllFound) {
    constexpr int N = 1000;
    insertRange(N, 10);
    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N));
    for (int i = 0; i < N; ++i) {
        auto res = tree->find(static_cast<BTreeKey>(i));
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i) * 10);
    }
}

TEST_F(BPlusTreeTest, InsertRandomOrderAllFound) {
    constexpr int N = 500;
    std::vector<BTreeKey> keys(N);
    std::iota(keys.begin(), keys.end(), 1);
    std::mt19937 rng(42);
    std::ranges::shuffle(keys, rng);
    for (auto k : keys) insertUint(k, k * 3);
    std::ranges::shuffle(keys, rng);
    for (auto k : keys) {
        auto res = tree->find(k);
        ASSERT_TRUE(res.has_value()) << "key=" << k;
        EXPECT_EQ(readUint(*res), k * 3);
    }
}

TEST_F(BPlusTreeTest, TreeGrowsInHeight) {
    insertRange(10'000);
    EXPECT_GT(tree->height(), 1u);
}

TEST_F(BPlusTreeTest, RangeScanReturnsCorrectSubset) {
    for (BTreeKey k = 0; k < 20; ++k) insertUint(k, k);
    std::vector<BTreeKey> found;
    tree->rangeScan(5, 10, [&](const BTreeEntry& e) {
        found.push_back(e.key);
        return true;
    });
    ASSERT_EQ(found.size(), 5u);
    EXPECT_EQ(found.front(), 5u);
    EXPECT_EQ(found.back(), 9u);
}

TEST_F(BPlusTreeTest, RangeScanIsAscending) {
    for (BTreeKey k : { 10u, 1u, 5u, 8u, 3u }) insertUint(k, k);
    std::vector<BTreeKey> found;
    tree->rangeScan(1, 11, [&](const BTreeEntry& e) {
        found.push_back(e.key);
        return true;
    });
    EXPECT_TRUE(std::ranges::is_sorted(found));
}

TEST_F(BPlusTreeTest, RangeScanCanBeInterrupted) {
    insertRange(100);
    int count = 0;
    tree->rangeScan(0, 100, [&](const BTreeEntry&) { return ++count < 5; });
    EXPECT_EQ(count, 5);
}

TEST_F(BPlusTreeTest, ScanAllReturnsAllInOrder) {
    for (BTreeKey k : { 3u, 1u, 4u, 5u, 9u, 2u, 6u }) insertUint(k, k);
    auto keys = collectKeys();
    EXPECT_TRUE(std::ranges::is_sorted(keys));
    EXPECT_EQ(std::ranges::adjacent_find(keys), keys.end());
}

TEST_F(BPlusTreeTest, RangeScanAcrossLeafBoundary) {
    insertRange(500);
    std::vector<BTreeKey> found;
    tree->rangeScan(100, 200, [&](const BTreeEntry& e) {
        found.push_back(e.key);
        return true;
    });
    ASSERT_EQ(found.size(), 100u);
    EXPECT_EQ(found.front(), 100u);
    EXPECT_EQ(found.back(), 199u);
    EXPECT_TRUE(std::ranges::is_sorted(found));
}

TEST_F(BPlusTreeTest, RedistributeLeaf_FromRightNeighbor) {
    constexpr int N = 200;
    insertRange(N);
    for (int i = 0; i < N / 2; ++i) tree->remove(i);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N / 2));
    for (int i = N / 2; i < N; ++i) EXPECT_TRUE(tree->find(i).has_value()) << "key=" << i;
    auto keys = collectKeys();
    EXPECT_TRUE(std::ranges::is_sorted(keys));
    EXPECT_EQ(keys.size(), static_cast<std::size_t>(N / 2));
}

TEST_F(BPlusTreeTest, RedistributeLeaf_FromLeftNeighbor) {
    constexpr int N = 200;
    insertRange(N);
    for (int i = N / 2; i < N; ++i) tree->remove(i);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N / 2));
    for (int i = 0; i < N / 2; ++i) EXPECT_TRUE(tree->find(i).has_value()) << "key=" << i;
    auto keys = collectKeys();
    EXPECT_TRUE(std::ranges::is_sorted(keys));
}

TEST_F(BPlusTreeTest, RedistributeLeaf_ValuesCorrectAfterRedistribute) {
    constexpr int N = 150;
    for (int i = 0; i < N; ++i) insertUint(i, i * 7);
    for (int i = N * 2 / 3; i < N; ++i) tree->remove(i);

    for (int i = 0; i < N * 2 / 3; ++i) {
        auto res = tree->find(i);
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i) * 7);
    }
}

TEST_F(BPlusTreeTest, MergeLeaf_HeightReducesAfterMassDelete) {
    constexpr int N = 1000;
    insertRange(N);
    uint32_t heightAfterInsert = tree->height();
    EXPECT_GT(heightAfterInsert, 1u);

    for (int i = 1; i < N; ++i) tree->remove(i);

    EXPECT_LT(tree->height(), heightAfterInsert);
    EXPECT_EQ(tree->size(), 1u);
    EXPECT_TRUE(tree->find(0).has_value());
}

TEST_F(BPlusTreeTest, MergeLeaf_AllKeysDeletedTreeIsEmpty) {
    constexpr int N = 500;
    insertRange(N);
    for (int i = 0; i < N; ++i) tree->remove(i);

    EXPECT_EQ(tree->size(), 0u);
    EXPECT_EQ(tree->height(), 1u);
    EXPECT_TRUE(collectKeys().empty());
}

TEST_F(BPlusTreeTest, MergeLeaf_RemainingValuesCorrect) {
    constexpr int N = 300;
    for (int i = 0; i < N; ++i) insertUint(i, i * 5);
    for (int i = 1; i < N; i += 2) tree->remove(i);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N / 2));
    for (int i = 0; i < N; i += 2) {
        auto res = tree->find(i);
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i) * 5);
    }
}

TEST_F(BPlusTreeTest, MergeLeaf_LinkedListConsistentAfterMerge) {
    constexpr int N = 400;
    insertRange(N);
    for (int i = 0; i < N; i += 3) tree->remove(i);

    auto keys = collectKeys();
    EXPECT_TRUE(std::ranges::is_sorted(keys));
    EXPECT_EQ(std::ranges::adjacent_find(keys), keys.end());
    for (auto k : keys) EXPECT_NE(k % 3, 0u) << "deleted key " << k << " found";
}

TEST_F(BPlusTreeTest, MergeLeaf_DeleteFromMiddle) {
    constexpr int N = 300;
    insertRange(N);
    int lo = N / 4, hi = N * 3 / 4;
    for (int i = lo; i < hi; ++i) tree->remove(i);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N - (hi - lo)));
    for (int i = 0; i < lo; ++i) EXPECT_TRUE(tree->find(i).has_value()) << "key=" << i;
    for (int i = lo; i < hi; ++i) EXPECT_FALSE(tree->find(i).has_value()) << "key=" << i;
    for (int i = hi; i < N; ++i) EXPECT_TRUE(tree->find(i).has_value()) << "key=" << i;
}

TEST_F(BPlusTreeTest, RedistributeInner_ScanCorrect) {
    constexpr int N = 27'000;
    insertRange(N);
    EXPECT_GE(tree->height(), 3u);

    for (int i = 0; i < N; i += 3) tree->remove(i);

    auto keys = collectKeys();
    EXPECT_TRUE(std::ranges::is_sorted(keys));
    EXPECT_EQ(std::ranges::adjacent_find(keys), keys.end());
    EXPECT_EQ(keys.size(), static_cast<std::size_t>(N - N / 3));
}

TEST_F(BPlusTreeTest, CollapseRoot_HeightDecreases) {
    constexpr int N = 5000;
    insertRange(N);
    EXPECT_GT(tree->height(), 1u);

    for (int i = 1; i < N; ++i) tree->remove(i);

    EXPECT_EQ(tree->height(), 1u);
    EXPECT_EQ(tree->size(), 1u);
    EXPECT_TRUE(tree->find(0).has_value());
}

TEST_F(BPlusTreeTest, CollapseRoot_SparseKeysRemain) {
    constexpr int N = 2000;
    for (int i = 0; i < N; ++i) insertUint(i, i * 3);
    for (int i = 0; i < N; ++i)
        if (i % 100 != 0)
            tree->remove(i);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N / 100));
    for (int i = 0; i < N; i += 100) {
        auto res = tree->find(i);
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i) * 3);
    }
}

TEST_F(BPlusTreeTest, InsertDeleteAllConsistency) {
    constexpr int N = 500;
    insertRange(N);
    for (int i = 0; i < N; ++i) tree->remove(i);

    EXPECT_EQ(tree->size(), 0u);
    EXPECT_EQ(tree->height(), 1u);
    for (int i = 0; i < N; ++i) EXPECT_FALSE(tree->find(i).has_value()) << "key=" << i;
}

TEST_F(BPlusTreeTest, RepeatedInsertDeleteCycles) {
    for (int cycle = 0; cycle < 5; ++cycle) {
        constexpr int N = 200;
        for (int i = 0; i < N; ++i) insertUint(i, i + cycle * 1000);
        for (int i = 0; i < N / 2; ++i) tree->remove(i);

        for (int i = N / 2; i < N; ++i) {
            auto res = tree->find(i);
            ASSERT_TRUE(res.has_value()) << "cycle=" << cycle << " key=" << i;
            EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i + cycle * 1000));
        }
        for (int i = N / 2; i < N; ++i) tree->remove(i);
        EXPECT_EQ(tree->size(), 0u) << "cycle=" << cycle;
    }
}

TEST_F(BPlusTreeTest, RandomInsertDeleteConsistency) {
    std::mt19937 rng(123);
    std::map<BTreeKey, uint64_t> reference;

    for (int op = 0; op < 2000; ++op) {
        BTreeKey key = rng() % 300;
        if (rng() % 3 != 0) {
            uint64_t val = rng();
            insertUint(key, val);
            reference[key] = val;
        } else {
            tree->remove(key);
            reference.erase(key);
        }
    }

    EXPECT_EQ(tree->size(), reference.size());
    for (auto& [key, val] : reference) {
        auto res = tree->find(key);
        ASSERT_TRUE(res.has_value()) << "key=" << key;
        EXPECT_EQ(readUint(*res), val);
    }

    auto keys = collectKeys();
    std::vector<BTreeKey> refKeys;
    for (auto& [k, v] : reference) refKeys.push_back(k);
    EXPECT_EQ(keys, refKeys);
}

TEST_F(BPlusTreeTest, DataPersistsAfterReopen) {
    constexpr int N = 200;
    for (int i = 0; i < N; ++i) insertUint(i, i * 7);
    pm->sync();

    tree.reset();
    pm = PageManager::open(dbPath);
    tree = BPlusTree::open(*pm, rootPageId);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N));
    for (int i = 0; i < N; ++i) {
        auto res = tree->find(i);
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i) * 7);
    }
}

TEST_F(BPlusTreeTest, PersistAfterMerge) {
    constexpr int N = 300;
    insertRange(N);
    for (int i = N / 2; i < N; ++i) tree->remove(i);
    pm->sync();

    tree.reset();
    pm = PageManager::open(dbPath);
    tree = BPlusTree::open(*pm, rootPageId);

    EXPECT_EQ(tree->size(), static_cast<uint64_t>(N / 2));
    for (int i = 0; i < N / 2; ++i) EXPECT_TRUE(tree->find(i).has_value()) << "key=" << i;

    auto keys = collectKeys();
    EXPECT_TRUE(std::ranges::is_sorted(keys));
    EXPECT_EQ(keys.size(), static_cast<std::size_t>(N / 2));
}
