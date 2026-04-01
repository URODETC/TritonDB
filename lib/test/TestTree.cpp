#include <filesystem>
#include <span>
#include <string>
#include <system_error>

#include "gtest/gtest.h"
#include "storage/PageManager.h"
#include "storage/Tree.h"

namespace fs = std::filesystem;
using tritondb::storage::PageManager;
using tritondb::storage::Tree;

namespace {
class TreeTest : public ::testing::Test {
protected:
    fs::path file_ { std::filesystem::temp_directory_path() / "test_tree.db" };

    void SetUp() override {
        std::error_code ec;
        fs::remove(file_, ec);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove(file_, ec);
    }
};
}   // namespace

TEST_F(TreeTest, EmptyTreeFindReturnsNullopt) {
    PageManager pm(file_);
    Tree tree(pm);

    EXPECT_FALSE(tree.find(42).has_value());
}

TEST_F(TreeTest, InsertAndFindSingleKey) {
    PageManager pm(file_);
    Tree tree(pm);

    const uint64_t key = 42;
    std::string data = "Hello, TritonDB!";
    const std::span<const std::byte> value(reinterpret_cast<const std::byte*>(data.data()), data.size());

    EXPECT_TRUE(tree.insert(key, value));

    const auto result = tree.find(key);

    ASSERT_TRUE(result.has_value());

    const std::string result_str(reinterpret_cast<const char*>(result->data()), result->size());
    EXPECT_EQ(result_str, data);
}