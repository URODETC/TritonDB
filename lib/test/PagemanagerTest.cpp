#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "storage/PageManager.h"

using namespace tritondb::storage;

class PageManagerTest : public ::testing::Test {
protected:
    std::filesystem::path dbPath_;

    void SetUp() override {
        dbPath_ = std::filesystem::temp_directory_path() / "tritondb_pm_test.db";
        std::filesystem::remove(dbPath_);
    }

    void TearDown() override { std::filesystem::remove(dbPath_); }
};

TEST_F(PageManagerTest, OpenCreatesFile) {
    auto pm = PageManager::open(dbPath_);
    ASSERT_TRUE(std::filesystem::exists(dbPath_));
}

TEST_F(PageManagerTest, AllocReturnsValidPageId) {
    auto pm = PageManager::open(dbPath_);
    PageId id = pm->alloc(PageType::BTreeLeaf);
    EXPECT_NE(id, kInvalidPageId);
}

TEST_F(PageManagerTest, AllocIncreasesPageCount) {
    auto pm = PageManager::open(dbPath_);
    uint64_t before = pm->pageCount();
    pm->alloc(PageType::BTreeLeaf);
    EXPECT_EQ(pm->pageCount(), before + 1);
}

TEST_F(PageManagerTest, AllocMultipleReturnsUniqueIds) {
    auto pm = PageManager::open(dbPath_);
    constexpr int N = 100;
    std::vector<PageId> ids;
    for (int i = 0; i < N; ++i) {
        ids.push_back(pm->alloc(PageType::BTreeLeaf));
    }
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(std::unique(ids.begin(), ids.end()), ids.end());
}

TEST_F(PageManagerTest, WriteAndReadRoundtrip) {
    auto pm = PageManager::open(dbPath_);
    PageId id = pm->alloc(PageType::BTreeLeaf);

    Page p = pm->read(id);
    p.data[0] = std::byte { 0xAB };
    p.data[1] = std::byte { 0xCD };
    pm->write(id, p);

    Page readBack = pm->read(id);
    EXPECT_EQ(readBack.data[0], std::byte { 0xAB });
    EXPECT_EQ(readBack.data[1], std::byte { 0xCD });
}

TEST_F(PageManagerTest, ReadPreservesPageType) {
    auto pm = PageManager::open(dbPath_);
    PageId id = pm->alloc(PageType::BTreeInner);

    Page p = pm->read(id);
    EXPECT_EQ(p.header.type, PageType::BTreeInner);
}

TEST_F(PageManagerTest, DataPersistsAfterReopen) {
    PageId id;
    {
        auto pm = PageManager::open(dbPath_);
        id = pm->alloc(PageType::BTreeLeaf);

        Page p = pm->read(id);
        p.data[42] = std::byte { 0xFF };
        pm->write(id, p);
        pm->sync();
    }
    {
        auto pm = PageManager::open(dbPath_);
        Page p = pm->read(id);
        EXPECT_EQ(p.data[42], std::byte { 0xFF });
    }
}

TEST_F(PageManagerTest, PageCountPersistsAfterReopen) {
    uint64_t count_before;
    {
        auto pm = PageManager::open(dbPath_);
        pm->alloc(PageType::BTreeLeaf);
        pm->alloc(PageType::BTreeLeaf);
        pm->sync();
        count_before = pm->pageCount();
    }
    {
        auto pm = PageManager::open(dbPath_);
        EXPECT_EQ(pm->pageCount(), count_before);
    }
}

TEST_F(PageManagerTest, FreedPageIsReused) {
    auto pm = PageManager::open(dbPath_);
    PageId id1 = pm->alloc(PageType::BTreeLeaf);
    pm->free(id1);
    PageId id2 = pm->alloc(PageType::BTreeLeaf);
    // Освобождённая страница должна быть переиспользована
    EXPECT_EQ(id1, id2);
}

TEST_F(PageManagerTest, FreeListPersistsAfterReopen) {
    PageId freed;
    {
        auto pm = PageManager::open(dbPath_);
        freed = pm->alloc(PageType::BTreeLeaf);
        pm->free(freed);
        pm->sync();
    }
    {
        auto pm = PageManager::open(dbPath_);
        PageId reused = pm->alloc(PageType::BTreeLeaf);
        EXPECT_EQ(freed, reused);
    }
}

TEST_F(PageManagerTest, ReadThrowsOnCorruptedChecksum) {
    auto pm = PageManager::open(dbPath_);
    PageId id = pm->alloc(PageType::BTreeLeaf);
    pm->sync();

    {
        std::fstream f(dbPath_, std::ios::in | std::ios::out | std::ios::binary);
        off_t offset = (static_cast<off_t>(id) * kPageSize) + kPageHeaderSize;
        f.seekp(offset);
        char garbage = 0xDE;
        f.write(&garbage, 1);
    }

    EXPECT_THROW(pm->read(id), std::runtime_error);
}