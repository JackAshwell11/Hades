// External includes
#include "gtest/gtest.h"

// Custom includes
#include "primitives.hpp"

// ----- FIXTURES ------------------------------
class Points : public testing::Test {
 protected:
  Point valid_point_one{3, 5}, valid_point_two{5, 7}, boundary_point{4, 0}, zero_point{0, 0};
};
