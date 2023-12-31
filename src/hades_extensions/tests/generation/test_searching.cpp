// Local headers
#include "generation/searching.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the generation/searching.hpp tests.
class SearchingFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  const Grid grid{6, 9};

  /// A position in the middle of the grid for use in testing.
  const Position position_one{3, 7};

  /// An extra position in the middle of the grid for use in testing.
  const Position position_two{4, 1};

  /// A position on the edge of the grid for use in testing.
  const Position position_three{4, 0};

  /// Add obstacles to the grid for use in testing.
  void add_obstacles() const {
    grid.set_value({1, 3}, TileType::Obstacle);
    grid.set_value({2, 7}, TileType::Obstacle);
    grid.set_value({3, 2}, TileType::Obstacle);
    grid.set_value({3, 3}, TileType::Obstacle);
    grid.set_value({3, 6}, TileType::Obstacle);
    grid.set_value({4, 3}, TileType::Obstacle);
    grid.set_value({4, 6}, TileType::Obstacle);
  }

  /// Add item and floor tiles to the grid for use in testing.
  ///
  /// @param items The positions of the items to add.
  /// @param all Whether to add item tiles to all positions.
  void add_items_and_floors(const std::unordered_set<Position> &items, const bool all = false) const {
    for (int y = 0; y < grid.height; y++) {
      for (int x = 0; x < grid.width; x++) {
        if (all) {
          grid.set_value({x, y}, TileType::Obstacle);
        } else {
          grid.set_value({x, y}, !items.contains({x, y}) ? TileType::Floor : TileType::Obstacle);
        }
      }
    }
  }
};

// ----- TESTS ------------------------------
/// Test that A* works in a grid with no obstacles when started in the middle.
TEST_F(SearchingFixture, TestCalculateAstarPathNoObstaclesMiddleStart) {
  const std::vector<Position> no_obstacles_result{{4, 1}, {3, 2}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), no_obstacles_result);
}

/// Test that A* works in a grid with no obstacles when ended on the edge.
TEST_F(SearchingFixture, TestCalculateAstarPathNoObstaclesBoundaryEnd) {
  const std::vector<Position> no_obstacles_result{{4, 0}, {3, 1}, {3, 2}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), no_obstacles_result);
}

/// Test that A* works in a grid with obstacles when started in the middle.
TEST_F(SearchingFixture, TestCalculateAstarPathObstaclesMiddleStart) {
  add_obstacles();
  const std::vector<Position> obstacles_result{{4, 1}, {4, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), obstacles_result);
}

/// Test that A* works in a grid with obstacles when ended on the edge.
TEST_F(SearchingFixture, TestCalculateAstarPathObstaclesBoundaryEnd) {
  add_obstacles();
  const std::vector<Position> obstacles_result{{4, 0}, {3, 1}, {2, 2}, {2, 3}, {3, 4}, {2, 5}, {2, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), obstacles_result);
}

/// Test that A* throws an exception in an empty grid.
TEST_F(SearchingFixture,
       TestCalculateAstarPathEmptyGrid){ASSERT_THROW_MESSAGE(calculate_astar_path({0, 0}, position_one, position_two),
                                                             std::length_error, "Grid size must be bigger than 0.")}

/// Test that generate_item_position returns a valid position within the minimum distance.
TEST_F(SearchingFixture, TestGenerateItemPositionWithinPosition) {
  add_items_and_floors({position_one});
  const auto position_result = generate_item_position(grid, {position_one}, true);
  ASSERT_TRUE(position_result != Position(-1, -1));
  const auto [diff_x, diff_y] = position_result - position_one;
  ASSERT_TRUE(diff_x <= 5 && diff_y <= 5);
}

/// Test that generate_item_position returns an invalid position if no valid positions are within the minimum distance.
TEST_F(SearchingFixture, TestGenerateItemPositionWithinNoValidPositions) {
  add_items_and_floors({}, true);
  ASSERT_EQ(generate_item_position(grid, {position_one}, true), Position(-1, -1));
}

/// Test that generate_item_position returns a valid position within the minimum distance if multiple valid item
/// positions are given.
TEST_F(SearchingFixture, TestGenerateItemPositionWithinMultipleItems) {
  add_items_and_floors({position_one, position_two, position_three});
  const auto position_result = generate_item_position(grid, {position_one, position_two, position_three}, true);
  ASSERT_TRUE(position_result != Position(-1, -1));
  const auto [diff_x, diff_y] = position_result - position_one;
  ASSERT_TRUE(diff_x <= 5 && diff_y <= 5);
}

/// Test that generate_item_position returns a valid position outside the minimum distance.
TEST_F(SearchingFixture, TestGenerateItemPositionOutsidePosition) {
  add_items_and_floors({position_one});
  const auto position_result = generate_item_position(grid, {position_one}, false);
  ASSERT_TRUE(position_result != Position(-1, -1));
  const auto [diff_x, diff_y] = position_result - position_one;
  ASSERT_TRUE(diff_x > 5 || diff_y > 5);
}

/// Test that generate_item_position returns an invalid position if no valid positions are outside the minimum distance.
TEST_F(SearchingFixture, TestGenerateItemPositionOutsideNoValidPositions) {
  add_items_and_floors({}, true);
  ASSERT_EQ(generate_item_position(grid, {position_one}, false), Position(-1, -1));
}

/// Test that generate_item_position returns a valid position outside the minimum distance if multiple valid item
/// positions are given.
TEST_F(SearchingFixture, TestGenerateItemPositionOutsideMultipleItems) {
  add_items_and_floors({position_one, position_two, position_three});
  ASSERT_EQ(generate_item_position(grid, {position_one, position_two, position_three}, false), Position(-1, -1));
}

/// Test that generate_item_position returns an invalid position if no item positions are given.
TEST_F(SearchingFixture, TestGenerateItemPositionEmptyItemPositions) {
  ASSERT_EQ(generate_item_position(grid, {}, false), Position(-1, -1));
}

/// Test that generate_item_position throws an exception in an empty grid.
TEST_F(SearchingFixture, TestGenerateItemPositionEmptyGrid) {
  ASSERT_THROW_MESSAGE(generate_item_position({0, 0}, {}, false), std::length_error, "Grid size must be bigger than 0.")
}
