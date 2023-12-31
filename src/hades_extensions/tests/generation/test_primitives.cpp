// Local headers
#include "generation/primitives.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the generation/primitives.hpp tests.
class PrimitivesFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  const Grid grid{5, 5};

  /// A rect inside the grid for use in testing.
  const Rect rect_one{{0, 0}, {2, 3}};

  /// A rect inside the grid for use in testing.
  const Rect rect_two{{2, 2}, {4, 4}};
};

// ----- TESTS ------------------------------
/// Test that finding the distance between two identical rects works correctly.
TEST_F(PrimitivesFixture, TestRectGetDistanceToIdentical) { ASSERT_EQ(rect_one.get_distance_to(rect_one), 0); }

/// Test that finding the distance between two different rects works correctly.
TEST_F(PrimitivesFixture, TestRectGetDistanceToDifferent) { ASSERT_EQ(rect_one.get_distance_to(rect_two), 2); }

/// Test that a position with two small values is not within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinSmallerXY) { ASSERT_FALSE(grid.is_position_within({-1, -1})); }

/// Test that a position with a small X value is not within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinSmallerX) { ASSERT_FALSE(grid.is_position_within({-1, 3})); }

/// Test that a position with a small Y value is not within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinSmallerY) { ASSERT_FALSE(grid.is_position_within({3, -1})); }

/// Test that a position with two lower boundary values is within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinLowerBoundaryXY) { ASSERT_TRUE(grid.is_position_within({0, 0})); }

/// Test that a position with a lower boundary X value is within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinLowerBoundaryX) { ASSERT_TRUE(grid.is_position_within({0, 3})); }

/// Test that a position with a lower boundary Y value is within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinLowerBoundaryY) { ASSERT_TRUE(grid.is_position_within({3, 0})); }

/// Test that a position with two upper boundary values is within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinUpperBoundaryXY) { ASSERT_TRUE(grid.is_position_within({4, 4})); }

/// Test that a position with an upper boundary X value is within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinUpperBoundaryX) { ASSERT_TRUE(grid.is_position_within({4, 3})); }

/// Test that a position with an upper boundary Y value is within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinUpperBoundaryY) { ASSERT_TRUE(grid.is_position_within({3, 4})); }

/// Test that a position with two large values is not within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinLargerXY) { ASSERT_FALSE(grid.is_position_within({5, 5})); }

/// Test that a position with a large X value is not within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinLargerX) { ASSERT_FALSE(grid.is_position_within({5, 3})); }

/// Test that a position with a large Y value is not within the grid.
TEST_F(PrimitivesFixture, TestGridIsPositionWithinLargerY) { ASSERT_FALSE(grid.is_position_within({3, 5})); }

/// Test that a position in the middle of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertPositionMiddle) { ASSERT_EQ(grid.convert_position({1, 2}), 11); }

/// Test that a position on the top of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertPositionEdgeTop) { ASSERT_EQ(grid.convert_position({3, 0}), 3); }

/// Test that a position on the bottom of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertPositionEdgeBottom) { ASSERT_EQ(grid.convert_position({2, 4}), 22); }

/// Test that a position on the left of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertPositionEdgeLeft) { ASSERT_EQ(grid.convert_position({0, 3}), 15); }

/// Test that a position on the right of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertPositionEdgeRight) { ASSERT_EQ(grid.convert_position({1, 4}), 21); }

/// Test that converting a position smaller than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridConvertPositionSmaller){
    ASSERT_THROW_MESSAGE((grid.convert_position({-1, -1})), std::out_of_range, "Position not within the grid.")}

/// Test that converting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridConvertPositionLarger){
    ASSERT_THROW_MESSAGE((grid.convert_position({10, 10})), std::out_of_range, "Position not within the grid.")}

/// Test that a position in the middle of the grid can be retrieved correctly.
TEST_F(PrimitivesFixture, TestGridGetValueMiddle) {
  (*grid.grid)[13] = TileType::Player;
  ASSERT_EQ(grid.get_value({3, 2}), TileType::Player);
}

/// Test that a position on the edge of the grid can be retrieved correctly.
TEST_F(PrimitivesFixture, TestGridGetValueEdge) {
  (*grid.grid)[23] = TileType::Player;
  ASSERT_EQ(grid.get_value({3, 4}), TileType::Player);
}

/// Test that getting a position smaller than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridGetValueSmaller){
    ASSERT_THROW_MESSAGE((grid.get_value({-1, -1})), std::out_of_range, "Position not within the grid.")}

/// Test that getting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridGetValueLarger){
    ASSERT_THROW_MESSAGE((grid.get_value({10, 10})), std::out_of_range, "Position not within the grid.")}

/// Test that a position in the middle can be set correctly.
TEST_F(PrimitivesFixture, TestGridSetValueMiddle) {
  grid.set_value({1, 3}, TileType::Player);
  ASSERT_EQ((*grid.grid)[16], TileType::Player);
}

/// Test that a position on the edge can be set correctly.
TEST_F(PrimitivesFixture, TestGridSetValueEdge) {
  grid.set_value({4, 4}, TileType::Player);
  ASSERT_EQ((*grid.grid)[24], TileType::Player);
}

/// Test that setting a position smaller than the array throws an exception.
TEST_F(PrimitivesFixture,
       TestGridSetValueSmaller){ASSERT_THROW_MESSAGE((grid.set_value({-1, -1}, TileType::Player)), std::out_of_range,
                                                     "Position not within the grid.")}

/// Test that setting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture,
       TestGridSetValueLarger){ASSERT_THROW_MESSAGE((grid.set_value({10, 10}, TileType::Player)), std::out_of_range,
                                                    "Position not within the grid.")}

/// Test that a zero size rect can be placed correctly in a valid grid.
TEST_F(PrimitivesFixture, TestGridPlaceRectZeroSize) {
  grid.place_rect({{0, 0}, {0, 0}});
  const std::vector target_result{
      TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, target_result);
}

/// Test that a rect can be placed correctly in a valid grid.
TEST_F(PrimitivesFixture, TestGridPlaceRectValidGrid) {
  grid.place_rect(rect_one);
  const std::vector target_result{
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, target_result);
}

/// Test that placing a rect that doesn't fit in the grid works correctly.
TEST_F(PrimitivesFixture, TestGridPlaceRectOutsideGrid) {
  grid.place_rect({{0, 0}, {10, 10}});
  const std::vector target_result{
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
  };
  ASSERT_EQ(*grid.grid, target_result);
}

/// Test that placing a rect in a zero size grid doesn't do anything.
TEST_F(PrimitivesFixture, TestGridPlaceRectZeroSizeGrid) {
  const Grid empty_grid{0, 0};
  empty_grid.place_rect(rect_one);
  ASSERT_EQ(*empty_grid.grid, std::vector<TileType>{});
}
