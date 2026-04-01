#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <system_error>

#include "storage/PageManager.h"

namespace fs = std::filesystem;
using tritondb::storage::kPageSize;
using tritondb::storage::Page;
using tritondb::storage::PageId;
using tritondb::storage::PageManager;

namespace {
fs::path make_temp_db_file(const std::string& suffix) {
    const auto base = fs::temp_directory_path();
    return base / ("tritiongb_pagemanager_" + suffix + ".db");
}

void remove_file_safely(const fs::path& file) {
    std::error_code ec;
    fs::remove(file, ec);
}
}   // namespace

TEST(PageManager, WriteThenReadSameBytes) {
    const auto file = make_temp_db_file("write_read");
    remove_file_safely(file);

    {
        PageManager pm(file);
        const PageId id = pm.alloc();

        Page out;
        out.id = id;
        for (std::size_t i = 0; i < kPageSize; ++i) {
            out.data[i] = static_cast<std::byte>(i % 251);
        }

        pm.write(out);
        const Page in = pm.read(id);

        EXPECT_EQ(in.id, out.id);
        EXPECT_EQ(in.data, out.data);
    }
    remove_file_safely(file);
}

TEST(PageManager, ReopenFileDataPersists) {
    const auto file = make_temp_db_file("reopen");
    remove_file_safely(file);

    PageId id {};
    {
        PageManager pm(file);
        id = pm.alloc();

        Page out;
        out.id = id;
        out.data[0] = std::byte { 0xAB };
        out.data[1] = std::byte { 0xCD };
        out.data[kPageSize - 1] = std::byte { 0xEF };

        pm.write(out);
    }

    {
        PageManager pm(file);
        const Page in = pm.read(id);

        EXPECT_EQ(in.data[0], std::byte { 0xAB });
        EXPECT_EQ(in.data[1], std::byte { 0xCD });
        EXPECT_EQ(in.data[kPageSize - 1], std::byte { 0xEF });
    }

    remove_file_safely(file);
}

TEST(PageManager, AllocReturnsUniqueIds) {
    const auto file = make_temp_db_file("alloc_unique");
    remove_file_safely(file);
    {
        PageManager pm(file);

        const PageId a = pm.alloc();
        const PageId b = pm.alloc();
        const PageId c = pm.alloc();

        EXPECT_NE(a, b);
        EXPECT_NE(b, c);
        EXPECT_NE(a, c);

        EXPECT_EQ(a, 0);
        EXPECT_EQ(b, 1);
        EXPECT_EQ(c, 2);
    }
    remove_file_safely(file);
}