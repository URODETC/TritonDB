#include <gtest/gtest.h>
#include "tritondb/tritondb.h"

TEST(Test, Case1) {
    tritondb g;
    EXPECT_EQ(g.hello(), 228);
}
