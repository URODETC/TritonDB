#include <gtest/gtest.h>
#include "storage/BPlusTree.h"
#include "storage/PageManager.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <random>

using namespace tritondb::storage;


class BPlusTreeTest : public ::testing::Test {
protected:
    std::filesystem::path dbPath;
    std::unique_ptr<PageManager> pm;
    std::unique_ptr<IBPlusTree>  tree;
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
        auto bytes = std::as_bytes(std::span{&val, 1});
        tree->insert(key, bytes);
    }

    uint64_t readUint(const BTreeValue& val) {
        uint64_t result = 0;
        std::memcpy(&result, val.data(), sizeof(result));
        return result;
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

TEST_F(BPlusTreeTest, FindMissingKeyReturnsNullopt) {
    EXPECT_FALSE(tree->find(999).has_value());
}

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

TEST_F(BPlusTreeTest, RemoveMissingKeyDoesNotThrow) {
    EXPECT_NO_THROW(tree->remove(999));
}


TEST_F(BPlusTreeTest, InsertManyKeysAllFound) {
    constexpr int N = 1000;
    for (int i = 0; i < N; ++i) {
        insertUint(static_cast<BTreeKey>(i), static_cast<uint64_t>(i * 10));
    }
    EXPECT_EQ(tree->size(), N);
    for (int i = 0; i < N; ++i) {
        auto res = tree->find(static_cast<BTreeKey>(i));
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i * 10));
    }
}

TEST_F(BPlusTreeTest, InsertRandomOrderAllFound) {
    constexpr int N = 500;
    std::vector<BTreeKey> keys(N);
    std::iota(keys.begin(), keys.end(), 1);

    std::mt19937 rng(42);
    std::shuffle(keys.begin(), keys.end(), rng);

    for (auto k : keys) insertUint(k, k * 3);

    std::shuffle(keys.begin(), keys.end(), rng);
    for (auto k : keys) {
        auto res = tree->find(k);
        ASSERT_TRUE(res.has_value()) << "key=" << k;
        EXPECT_EQ(readUint(*res), k * 3);
    }
}

TEST_F(BPlusTreeTest, TreeGrowsInHeight) {
    uint32_t initialHeight = tree->height();
    constexpr int N = 10'000;
    for (int i = 0; i < N; ++i) {
        insertUint(static_cast<BTreeKey>(i), 0);
    }
    EXPECT_GT(tree->height(), initialHeight);
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
    EXPECT_EQ(found.back(),  9u);
}

TEST_F(BPlusTreeTest, RangeScanIsAscending) {
    for (BTreeKey k : {10u, 1u, 5u, 8u, 3u}) insertUint(k, k);

    std::vector<BTreeKey> found;
    tree->rangeScan(1, 11, [&](const BTreeEntry& e) {
        found.push_back(e.key);
        return true;
    });
    EXPECT_TRUE(std::is_sorted(found.begin(), found.end()));
}

TEST_F(BPlusTreeTest, RangeScanCanBeInterrupted) {
    for (BTreeKey k = 0; k < 100; ++k) insertUint(k, k);

    int count = 0;
    tree->rangeScan(0, 100, [&](const BTreeEntry&) {
        ++count;
        return count < 5;
    });
    EXPECT_EQ(count, 5);
}

TEST_F(BPlusTreeTest, ScanAllReturnsAllInOrder) {
    for (BTreeKey k : {3u, 1u, 4u, 1u, 5u, 9u, 2u, 6u}) insertUint(k, k);

    std::vector<BTreeKey> found;
    tree->scan([&](const BTreeEntry& e) {
        found.push_back(e.key);
        return true;
    });
    EXPECT_TRUE(std::is_sorted(found.begin(), found.end()));
    EXPECT_EQ(std::adjacent_find(found.begin(), found.end()), found.end());
}


TEST_F(BPlusTreeTest, DataPersistsAfterReopen) {
    constexpr int N = 200;
    for (int i = 0; i < N; ++i) insertUint(static_cast<BTreeKey>(i), i * 7);
    pm->sync();

    tree.reset();
    pm = PageManager::open(dbPath);
    tree = BPlusTree::open(*pm, rootPageId);

    EXPECT_EQ(tree->size(), N);
    for (int i = 0; i < N; ++i) {
        auto res = tree->find(static_cast<BTreeKey>(i));
        ASSERT_TRUE(res.has_value()) << "key=" << i;
        EXPECT_EQ(readUint(*res), static_cast<uint64_t>(i * 7));
    }
}


TEST_F(BPlusTreeTest, DeleteManyKeysTreeStaysConsistent) {
    constexpr int N = 500;
    for (int i = 0; i < N; ++i) insertUint(i, i);

    for (int i = 0; i < N; i += 2) tree->remove(i);

    EXPECT_EQ(tree->size(), N / 2);
    for (int i = 0; i < N; ++i) {
        auto res = tree->find(i);
        if (i % 2 == 0) {
            EXPECT_FALSE(res.has_value()) << "deleted key " << i << " still found";
        } else {
            EXPECT_TRUE(res.has_value()) << "existing key " << i << " not found";
        }
    }
}