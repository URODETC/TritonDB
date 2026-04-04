#include <gtest/gtest.h>
#include "storage/PageManager.h"

#include <filesystem>
#include <fstream>

using namespace tritondb::storage;

class PageManagerTest : public ::testing::Test {
protected:
    std::filesystem::path dbPath;

    void SetUp() override {
        dbPath = std::filesystem::temp_directory_path() / "tritondb_pm_test.db";
        std::filesystem::remove(dbPath); // чистый старт
    }

    void TearDown() override {
        std::filesystem::remove(dbPath);
    }
};

TEST_F(PageManagerTest, OpenCreatesFile) {
    auto pm = PageManager::open(dbPath);
    ASSERT_TRUE(std::filesystem::exists(dbPath));
}

TEST_F(PageManagerTest, AllocReturnsValidPageId) {
    auto pm = PageManager::open(dbPath);
    PageId id = pm->alloc(PageType::TreeLeaf);
    EXPECT_NE(id, kInvalidPageId);
}

TEST_F(PageManagerTest, AllocIncreasesPageCount) {
    auto pm = PageManager::open(dbPath);
    uint64_t before = pm->pageCount();
    pm->alloc(PageType::TreeLeaf);
    EXPECT_EQ(pm->pageCount(), before + 1);
}

TEST_F(PageManagerTest, AllocMultipleReturnsUniqueIds) {
    auto pm = PageManager::open(dbPath);
    constexpr int N = 100;
    std::vector<PageId> ids;
    for (int i = 0; i < N; ++i) {
        ids.push_back(pm->alloc(PageType::TreeLeaf));
    }
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(std::unique(ids.begin(), ids.end()), ids.end());
}

TEST_F(PageManagerTest, WriteAndReadRoundtrip) {
    auto pm = PageManager::open(dbPath);
    PageId id = pm->alloc(PageType::TreeLeaf);

    Page p = pm->read(id);
    p.data[0] = std::byte{0xAB};
    p.data[1] = std::byte{0xCD};
    pm->write(id, p);

    Page readBack = pm->read(id);
    EXPECT_EQ(readBack.data[0], std::byte{0xAB});
    EXPECT_EQ(readBack.data[1], std::byte{0xCD});
}

TEST_F(PageManagerTest, ReadPreservesPageType) {
    auto pm = PageManager::open(dbPath);
    PageId id = pm->alloc(PageType::TreeInner);

    Page p = pm->read(id);
    EXPECT_EQ(p.header.type, PageType::TreeInner);
}

TEST_F(PageManagerTest, DataPersistsAfterReopen) {
    PageId id;
    {
        auto pm = PageManager::open(dbPath);
        id = pm->alloc(PageType::TreeLeaf);

        Page p = pm->read(id);
        p.data[42] = std::byte{0xFF};
        pm->write(id, p);
        pm->sync();
    }
    {
        auto pm = PageManager::open(dbPath);
        Page p = pm->read(id);
        EXPECT_EQ(p.data[42], std::byte{0xFF});
    }
}

TEST_F(PageManagerTest, PageCountPersistsAfterReopen) {
    uint64_t countBefore;
    {
        auto pm = PageManager::open(dbPath);
        pm->alloc(PageType::TreeLeaf);
        pm->alloc(PageType::TreeLeaf);
        pm->sync();
        countBefore = pm->pageCount();
    }
    {
        auto pm = PageManager::open(dbPath);
        EXPECT_EQ(pm->pageCount(), countBefore);
    }
}

TEST_F(PageManagerTest, FreedPageIsReused) {
    auto pm = PageManager::open(dbPath);
    PageId id1 = pm->alloc(PageType::TreeLeaf);
    pm->free(id1);
    PageId id2 = pm->alloc(PageType::TreeLeaf);
    // Освобождённая страница должна быть переиспользована
    EXPECT_EQ(id1, id2);
}

TEST_F(PageManagerTest, FreeListPersistsAfterReopen) {
    PageId freed;
    {
        auto pm = PageManager::open(dbPath);
        freed = pm->alloc(PageType::TreeLeaf);
        pm->free(freed);
        pm->sync();
    }
    {
        auto pm = PageManager::open(dbPath);
        PageId reused = pm->alloc(PageType::TreeLeaf);
        EXPECT_EQ(freed, reused);
    }
}

TEST_F(PageManagerTest, ReadThrowsOnCorruptedChecksum) {
    auto pm = PageManager::open(dbPath);
    PageId id = pm->alloc(PageType::TreeLeaf);
    pm->sync();

    {
        std::fstream f(dbPath, std::ios::in | std::ios::out | std::ios::binary);
        off_t offset = static_cast<off_t>(id) * kPageSize + kPageHeaderSize;
        f.seekp(offset);
        char garbage = 0xDE;
        f.write(&garbage, 1);
    }

    EXPECT_THROW(pm->read(id), std::runtime_error);
}