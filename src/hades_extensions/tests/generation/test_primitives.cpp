// Local headers
#include "generation/primitives.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the generation/primitives.hpp tests.
class PrimitivesFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  Grid grid{5, 5};

  /// A rect inside the grid for use in testing.
  const Rect rect_one{{0, 0}, {2, 3}};

  /// A rect inside the grid for use in testing.
  const Rect rect_two{{2, 2}, {4, 4}};
};

// ----- TESTS ------------------------------
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
    ASSERT_THROW_MESSAGE((grid.convert_position({-1, -1})), std::out_of_range, "Position must be within range")}

/// Test that converting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridConvertPositionLarger){
    ASSERT_THROW_MESSAGE((grid.convert_position({10, 10})), std::out_of_range, "Position must be within range")}

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
    ASSERT_THROW_MESSAGE((grid.get_value({-1, -1})), std::out_of_range, "Position must be within range")}

/// Test that getting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture, TestGridGetValueLarger){
    ASSERT_THROW_MESSAGE((grid.get_value({10, 10})), std::out_of_range, "Position must be within range")}

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
                                                     "Position must be within range")}

/// Test that setting a position larger than the array throws an exception.
TEST_F(PrimitivesFixture,
       TestGridSetValueLarger){ASSERT_THROW_MESSAGE((grid.set_value({10, 10}, TileType::Player)), std::out_of_range,
                                                    "Position must be within range")}

/// Test that finding the distance between two identical rects works correctly.
TEST_F(PrimitivesFixture, TestRectGetDistanceToIdentical) {
  ASSERT_EQ(rect_one.get_distance_to(rect_one), 0);
}

/// Test that finding the distance between two different rects works correctly.
TEST_F(PrimitivesFixture, TestRectGetDistanceToDifferent) { ASSERT_EQ(rect_one.get_distance_to(rect_two), 2); }

/// Test that a rect can be placed correctly in a valid grid.
TEST_F(PrimitivesFixture, TestRectPlaceRectValidGrid) {
  rect_one.place_rect(grid);
  const std::vector<TileType> target_result{
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Floor, TileType::Wall,  TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Floor, TileType::Wall,  TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, target_result);
}

/// Test that placing a rect that doesn't fit in the grid works correctly.
TEST_F(PrimitivesFixture, TestRectPlaceRectOutsideGrid) {
  const Rect invalid_rect{{0, 0}, {10, 10}};
  invalid_rect.place_rect(grid);
  const std::vector<TileType> target_result{
      TileType::Wall, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
  };
  ASSERT_EQ(*grid.grid, target_result);
}
