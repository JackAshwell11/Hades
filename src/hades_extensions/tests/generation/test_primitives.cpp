// Local headers
#include "generation/primitives.hpp"
#include "macros.hpp"

/// Implements the fixture for the generation/primitives.hpp tests.
class PrimitivesFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  Grid grid{5, 5};

  /// A rect inside the grid for use in testing.
  const Rect rect_one{{.x = 0, .y = 0}, {.x = 2, .y = 3}};

  /// A rect inside the grid for use in testing.
  const Rect rect_two{{.x = 2, .y = 2}, {.x = 4, .y = 4}};
};

/// Test that finding the distance between two identical positions works correctly.
TEST_F(PrimitivesFixture, TestPositionGetDistanceToIdentical) {
  ASSERT_EQ(rect_one.centre.get_distance_to(rect_one.centre), 0);
}

/// Test that finding the distance between two different positions works correctly.
TEST_F(PrimitivesFixture, TestPositionGetDistanceToDifferent) {
  ASSERT_EQ(rect_one.centre.get_distance_to(rect_two.centre), 2);
}

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

/// Test that an index at the start of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertIndexStart) {
  constexpr Position start_position{.x = 0, .y = 0};
  ASSERT_EQ(grid.convert_position(0), start_position);
}

/// Test that an index in the middle of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertIndexMiddle) {
  constexpr Position middle_position{.x = 1, .y = 2};
  ASSERT_EQ(grid.convert_position(11), middle_position);
}

/// Test that an index on the end of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertIndexEnd) {
  constexpr Position end_position{.x = 4, .y = 4};
  ASSERT_EQ(grid.convert_position(24), end_position);
}

/// Test that converting an index smaller than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridConvertIndexSmaller){
    ASSERT_THROW_MESSAGE((grid.convert_position(-1)), std::out_of_range, "Position not within the grid.")}

/// Test that converting an index larger than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridConvertIndexLarger){
    ASSERT_THROW_MESSAGE((grid.convert_position(100)), std::out_of_range, "Position not within the grid.")}

/// Test that a position in the middle of the grid can be converted correctly.
TEST_F(PrimitivesFixture, TestGridConvertPositionMiddle) {
  ASSERT_EQ(grid.convert_position({1, 2}), 11);
}

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
  grid.grid[13] = GameObjectType::Player;
  ASSERT_EQ(grid.get_value({3, 2}), GameObjectType::Player);
}

/// Test that a position on the edge of the grid can be retrieved correctly.
TEST_F(PrimitivesFixture, TestGridGetValueEdge) {
  grid.grid[23] = GameObjectType::Player;
  ASSERT_EQ(grid.get_value({3, 4}), GameObjectType::Player);
}

/// Test that getting a position smaller than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridGetValueSmaller){
    ASSERT_THROW_MESSAGE((grid.get_value({-1, -1})), std::out_of_range, "Position not within the grid.")}

/// Test that getting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridGetValueLarger){
    ASSERT_THROW_MESSAGE((grid.get_value({10, 10})), std::out_of_range, "Position not within the grid.")}

/// Test that a position in the middle can be set correctly.
TEST_F(PrimitivesFixture, TestGridSetValueMiddle) {
  grid.set_value({.x = 1, .y = 3}, GameObjectType::Player);
  ASSERT_EQ(grid.grid[16], GameObjectType::Player);
}

/// Test that a position on the edge can be set correctly.
TEST_F(PrimitivesFixture, TestGridSetValueEdge) {
  grid.set_value({.x = 4, .y = 4}, GameObjectType::Player);
  ASSERT_EQ(grid.grid[24], GameObjectType::Player);
}

/// Test that setting a position smaller than the array throws an exception.
TEST_F(PrimitivesFixture,
       TestGridSetValueSmaller){ASSERT_THROW_MESSAGE((grid.set_value({-1, -1}, GameObjectType::Player)),
                                                     std::out_of_range, "Position not within the grid.")}

/// Test that setting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture,
       TestGridSetValueLarger){ASSERT_THROW_MESSAGE((grid.set_value({10, 10}, GameObjectType::Player)),
                                                    std::out_of_range, "Position not within the grid.")}

/// Test that a zero size rect can be placed correctly in a valid grid.
TEST_F(PrimitivesFixture, TestGridPlaceRectZeroSize) {
  grid.place_rect({{.x = 0, .y = 0}, {.x = 0, .y = 0}});
  const std::vector target_result{
      GameObjectType::Floor, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty,
  };
  ASSERT_EQ(grid.grid, target_result);
}

/// Test that a rect can be placed correctly in a valid grid.
TEST_F(PrimitivesFixture, TestGridPlaceRectValidGrid) {
  grid.place_rect(rect_one);
  const std::vector target_result{
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Empty, GameObjectType::Empty,
      GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty, GameObjectType::Empty,
  };
  ASSERT_EQ(grid.grid, target_result);
}

/// Test that placing a rect that doesn't fit in the grid works correctly.
TEST_F(PrimitivesFixture, TestGridPlaceRectOutsideGrid) {
  grid.place_rect({{.x = 0, .y = 0}, {.x = 10, .y = 10}});
  const std::vector target_result{
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor,
      GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor, GameObjectType::Floor,
  };
  ASSERT_EQ(grid.grid, target_result);
}

/// Test that placing a rect in a zero size grid doesn't do anything.
TEST_F(PrimitivesFixture, TestGridPlaceRectZeroSizeGrid) {
  Grid empty_grid{0, 0};
  empty_grid.place_rect(rect_one);
  ASSERT_EQ(empty_grid.grid, std::vector<GameObjectType>{});
}
