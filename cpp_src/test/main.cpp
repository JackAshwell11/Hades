// External includes
#include "gtest/gtest.h"

TEST(hadesExtensionsTests, Example) {
  ASSERT_EQ(5*5, 25);
}

/// Run all the test in every file.
///
/// Parameters
/// ----------
/// argc - TODO
/// argv - TODO
///
/// Returns
/// -------
/// The result of all the test.
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
