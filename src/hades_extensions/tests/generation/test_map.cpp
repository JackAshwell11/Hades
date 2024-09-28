// Std headers
#include <algorithm>
#include <numeric>

// Local headers
#include "generation/map.hpp"
#include "macros.hpp"

/// Implements the fixture for the generation/map.hpp tests.
class MapFixture : public testing::Test {  // NOLINT
 protected:
  /// A random generator for use in testing.
  std::mt19937 random_generator;

  /// An 2D grid for use in testing.
  Grid grid{5, 5};

  /// A large 2D grid for use in testing.
  const Grid large_grid{8, 8};

  /// A very large 2D grid for use in testing.
  const Grid very_large_grid{25, 25};

  /// A rect that fits inside the grid for use in testing.
  const Rect rect_one{{.x = 0, .y = 1}, {.x = 3, .y = 4}};

  /// An extra rect that fits inside the grid for use in testing.
  const Rect rect_two{{.x = 2, .y = 1}, {.x = 4, .y = 2}};

  /// A large rect that doesn't fit inside the grid for use in testing.
  const Rect rect_three{{.x = 4, .y = 4}, {.x = 6, .y = 6}};

  /// Set up the fixture for the tests.
  void SetUp() override { random_generator.seed(0); }

  /// Place a rect made up of walls and floors in the grid for use in testing.
  void place_covered_box() const {
    very_large_grid.place_rect({{.x = 2, .y = 2}, {.x = 20, .y = 20}});
    for (int y = 1; y <= 21; y++) {
      for (int x = 1; x <= 21; x++) {
        if (very_large_grid.get_value({.x = x, .y = y}) != TileType::Floor) {
          very_large_grid.set_value({.x = x, .y = y}, TileType::Wall);
        }
      }
    }
  }
};

namespace {
/// Assert that there are no adjacent walls to the specified tile type in the grid.
///
/// @param grid - The grid to check for adjacent walls.
void assert_no_adjacent_walls(const Grid &grid) {
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      if (grid.get_value({.x = x, .y = y}) == TileType::HealthPotion) {
        const auto neighbours{grid.get_neighbours({.x = x, .y = y})};
        ASSERT_EQ(std::ranges::count_if(neighbours.begin(), neighbours.end(),
                                        [&grid](const auto &pos) { return grid.get_value(pos) == TileType::Wall; }),
                  0);
      }
    }
  }
}

/// Assert that each tile_type is a minimum distance away from every other tile_type in the grid.
///
/// @param grid - The grid to check for minimum distances.
/// @param tile_type - The tile type to check for minimum distances.
void assert_min_distance(const Grid &grid, const TileType tile_type) {
  // Collect all positions of the specified tile type
  std::vector<Position> positions;
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      if (grid.get_value({.x = x, .y = y}) == tile_type) {
        positions.emplace_back(x, y);
      }
    }
  }

  // Check that each position is at least min_distance away from every other position
  for (const auto start_pos : positions) {
    for (const auto end_pos : positions) {
      if (start_pos != end_pos) {
        const auto [x, y]{start_pos - end_pos};
        ASSERT_GE(std::min(x, y), 5);
      }
    }
  }
}
}  // namespace

/// Test that placing a tile in an empty grid does nothing.
TEST_F(MapFixture, TestMapPlaceTilesEmptyGrid) {
  const Grid empty_grid{0, 0};
  place_tiles(empty_grid, random_generator, TileType::Obstacle, 1, 1);
  ASSERT_EQ(*empty_grid.grid, std::vector<TileType>{});
}

/// Test that placing zero tiles in the grid does nothing.
TEST_F(MapFixture, TestMapPlaceTilesZeroCount) {
  place_tiles(grid, random_generator, TileType::Obstacle, 1, 0);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Empty), 25);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 0);
}

/// Test that placing an obstacle tile in the grid works correctly.
TEST_F(MapFixture, TestMapPlaceTilesObstacleSingleCount) {
  place_tiles(grid, random_generator, TileType::Obstacle, 1, 1);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Empty), 24);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 1);
  assert_min_distance(grid, TileType::Obstacle);
}

/// Test that placing an item tile in the grid works correctly.
TEST_F(MapFixture, TestMapPlaceTilesItemSingleCount) {
  place_covered_box();
  place_tiles(very_large_grid, random_generator, TileType::HealthPotion, 1, 1);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Floor), 360);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::HealthPotion), 1);
  assert_min_distance(very_large_grid, TileType::HealthPotion);
  assert_no_adjacent_walls(very_large_grid);
}

/// Test that placing multiple obstacle tiles in the grid works correctly.
TEST_F(MapFixture, TestMapPlaceTilesObstacleMultipleCount) {
  place_tiles(very_large_grid, random_generator, TileType::Obstacle, 1, 2);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Obstacle), 2);
  assert_min_distance(grid, TileType::Obstacle);
}

/// Test that placing multiple item tiles in the grid works correctly.
TEST_F(MapFixture, TestMapPlaceTilesItemMultipleCount) {
  place_covered_box();
  place_tiles(very_large_grid, random_generator, TileType::HealthPotion, 1, 2);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Floor), 359);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::HealthPotion), 2);
  assert_min_distance(very_large_grid, TileType::HealthPotion);
  assert_no_adjacent_walls(very_large_grid);
}

/// Test that placing an unknown number of obstacle tiles in the grid works correctly.
TEST_F(MapFixture, TestMapPlaceTilesObstacleUnknownCount) {
  place_tiles(very_large_grid, random_generator, TileType::Obstacle, 1, std::numeric_limits<int>::max());
  ASSERT_GE(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Empty), 621);
  ASSERT_GE(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Obstacle), 4);
  assert_min_distance(very_large_grid, TileType::Obstacle);
}

/// Test that placing an unknown number of item tiles in the grid works correctly.
TEST_F(MapFixture, TestMapPlaceTilesItemUnknownCount) {
  place_covered_box();
  place_tiles(very_large_grid, random_generator, TileType::HealthPotion, 1, std::numeric_limits<int>::max());
  ASSERT_GE(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Floor), 358);
  ASSERT_GE(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::HealthPotion), 2);
  assert_min_distance(very_large_grid, TileType::HealthPotion);
  assert_no_adjacent_walls(very_large_grid);
}

/// Test that placing a tile in the grid with a given probability works correctly.
TEST_F(MapFixture, TestMapPlaceTilesWithProbability) {
  place_tiles(grid, random_generator, TileType::Obstacle, 0.1, 1);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Empty), 25);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 0);
  place_tiles(grid, random_generator, TileType::Obstacle, 0.9, 1);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Empty), 24);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 1);
}

/// Test that placing a tile in the grid with no available positions throws an exception.
TEST_F(MapFixture, TestMapPlaceTilesNoAvailablePositions) {
  place_tiles(grid, random_generator, TileType::Floor, 1, 1);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Empty), 25);
}

/// Test that creating a minimum spanning tree with a single room works correctly.
TEST_F(MapFixture, TestMapCreateConnectionsSingleRoom) {
  const std::unordered_set<Edge> single_room_result{};
  ASSERT_EQ(create_connections({rect_one}), single_room_result);
}

/// Test that creating a minimum spanning tree with multiple rooms works correctly.
TEST_F(MapFixture, TestMapCreateConnectionsMultipleRooms) {
  // Create the minimum-spanning tree and check its size
  auto connections{create_connections({rect_one, rect_two, rect_three})};
  ASSERT_EQ(connections.size(), 2);

  // Check that the minimum spanning tree has the correct total cost
  ASSERT_EQ(std::accumulate(connections.begin(), connections.end(), 0,
                            [](const int &sum, const Edge &edge) { return sum + edge.cost; }),
            4);

  // Check that all the edges are correct
  for (const auto &edge : {Edge{.cost = 1, .source = rect_one, .destination = rect_two},
                           Edge{.cost = 3, .source = rect_one, .destination = rect_three}}) {
    ASSERT_TRUE(std::ranges::any_of(connections.begin(), connections.end(),
                                    [&edge](const Edge &connection) { return connection == edge; }));
  }
}

/// Test that creating a minimum spanning tree with zero rooms throws an exception.
TEST_F(MapFixture, TestMapCreateConnectionsEmptyRooms){
    ASSERT_THROW_MESSAGE(create_connections({}), std::length_error, "Rooms size must be bigger than 0.")}

/// Test that creating hallways with a single connection works correctly.
TEST_F(MapFixture, TestMapCreateHallwaysSingleConnection) {
  create_hallways(large_grid, {{.cost = 0, .source = rect_one, .destination = rect_three}});
  const std::vector single_connection_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*large_grid.grid, single_connection_result);
}

/// Test that creating hallways with multiple connections works correctly.
TEST_F(MapFixture, TestMapCreateHallwaysMultipleConnections) {
  create_hallways(large_grid, {{.cost = 0, .source = rect_one, .destination = rect_two},
                               {.cost = 0, .source = rect_one, .destination = rect_three}});
  const std::vector obstacles_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*large_grid.grid, obstacles_result);
}

/// Test that creating hallways with no connections doesn't do anything.
TEST_F(MapFixture, TestMapCreateHallwaysNoConnections) {
  create_hallways(large_grid, {});
  const std::vector no_connections_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*large_grid.grid, no_connections_result);
}

/// Test that running cellular automata on a grid with all empty tiles doesn't do anything.
TEST_F(MapFixture, TestMapRunCellularAutomataAllEmpty) {
  run_cellular_automata(grid);
  ASSERT_EQ(*grid.grid, std::vector(static_cast<std::size_t>(grid.width * grid.height), TileType::Empty));
}

/// Test that running cellular automata on a grid with all floor tiles sets the edges to walls.
TEST_F(MapFixture, TestMapRunCellularAutomataAllFloors) {
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      grid.set_value({.x = x, .y = y}, TileType::Floor);
    }
  }
  run_cellular_automata(grid);
  const std::vector all_floor_result{
      TileType::Wall, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
  };
  ASSERT_EQ(*grid.grid, all_floor_result);
}

/// Test that running cellular automata on a grid with all wall tiles sets all tiles to empty.
TEST_F(MapFixture, TestMapRunCellularAutomataAllWalls) {
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      grid.set_value({.x = x, .y = y}, TileType::Wall);
    }
  }
  run_cellular_automata(grid);
  ASSERT_EQ(*grid.grid, std::vector(static_cast<std::size_t>(grid.width * grid.height), TileType::Empty));
}

/// Test that running cellular automata on a grid with mixed floor and wall tiles works correctly.
TEST_F(MapFixture, TestMapRunCellularAutomataMixedFloors) {
  grid.set_value({.x = 1, .y = 1}, TileType::Floor);
  grid.set_value({.x = 2, .y = 1}, TileType::Floor);
  grid.set_value({.x = 3, .y = 1}, TileType::Floor);
  grid.set_value({.x = 1, .y = 2}, TileType::Floor);
  grid.set_value({.x = 3, .y = 2}, TileType::Floor);
  grid.set_value({.x = 1, .y = 3}, TileType::Floor);
  grid.set_value({.x = 2, .y = 3}, TileType::Floor);
  grid.set_value({.x = 3, .y = 3}, TileType::Floor);
  run_cellular_automata(grid);
  const std::vector mixed_floor_result{
      TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty,
      TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Wall,  TileType::Wall,
      TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, mixed_floor_result);
}

/// Test that running cellular automata on a grid with mixed floor and wall tiles works correctly.
TEST_F(MapFixture, TestMapRunCellularAutomataFloorsAtEdge) {
  grid.set_value({.x = 1, .y = 0}, TileType::Floor);
  grid.set_value({.x = 2, .y = 0}, TileType::Floor);
  grid.set_value({.x = 3, .y = 0}, TileType::Floor);
  grid.set_value({.x = 0, .y = 1}, TileType::Floor);
  grid.set_value({.x = 0, .y = 2}, TileType::Floor);
  grid.set_value({.x = 0, .y = 3}, TileType::Floor);
  grid.set_value({.x = 4, .y = 1}, TileType::Floor);
  grid.set_value({.x = 4, .y = 2}, TileType::Floor);
  grid.set_value({.x = 4, .y = 3}, TileType::Floor);
  grid.set_value({.x = 1, .y = 4}, TileType::Floor);
  grid.set_value({.x = 2, .y = 4}, TileType::Floor);
  grid.set_value({.x = 3, .y = 4}, TileType::Floor);
  run_cellular_automata(grid);
  const std::vector edge_floor_result{
      TileType::Wall, TileType::Wall,  TileType::Wall, TileType::Wall,  TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Wall, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Wall,  TileType::Wall, TileType::Wall,  TileType::Wall,
      TileType::Wall, TileType::Floor, TileType::Wall, TileType::Floor, TileType::Wall,
      TileType::Wall, TileType::Wall,  TileType::Wall, TileType::Wall,  TileType::Wall,
  };
  ASSERT_EQ(*grid.grid, edge_floor_result);
}

/// Test that running multiple cellular automata simulations works correctly.
TEST_F(MapFixture, TestMapRunCellularAutomataMultipleSimulations) {
  grid.set_value({.x = 0, .y = 0}, TileType::Floor);
  grid.set_value({.x = 2, .y = 0}, TileType::Floor);
  grid.set_value({.x = 4, .y = 0}, TileType::Floor);
  grid.set_value({.x = 0, .y = 2}, TileType::Floor);
  grid.set_value({.x = 2, .y = 1}, TileType::Floor);
  grid.set_value({.x = 1, .y = 2}, TileType::Floor);
  grid.set_value({.x = 2, .y = 2}, TileType::Floor);
  grid.set_value({.x = 3, .y = 2}, TileType::Floor);
  grid.set_value({.x = 4, .y = 2}, TileType::Floor);
  grid.set_value({.x = 2, .y = 3}, TileType::Floor);
  grid.set_value({.x = 0, .y = 4}, TileType::Floor);
  grid.set_value({.x = 2, .y = 4}, TileType::Floor);
  grid.set_value({.x = 4, .y = 4}, TileType::Floor);
  grid.set_value({.x = 4, .y = 4}, TileType::Floor);
  for (int i = 0; i < 3; i++) {
    run_cellular_automata(grid);
  }
  const std::vector multiple_simulation_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Wall,  TileType::Floor, TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, multiple_simulation_result);
}

/// Test that running cellular automata on an empty grid doesn't do anything.
TEST_F(MapFixture, TestMapRunCellularAutomataEmptyGrid) {
  Grid empty_grid{0, 0};
  run_cellular_automata(empty_grid);
  ASSERT_EQ(*empty_grid.grid, std::vector<TileType>{});
}

/// Test that creating a map with a valid level and seed works correctly.
TEST_F(MapFixture, TestMapCreateMapValidLevelSeed) {
  const auto [create_map_valid_grid, create_map_valid_constants] = create_map(0, 10);
  ASSERT_EQ(create_map_valid_constants.level, 0);
  ASSERT_EQ(create_map_valid_constants.width, 30);
  ASSERT_EQ(create_map_valid_constants.height, 20);
  ASSERT_EQ(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Player), 1);
  ASSERT_GE(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::HealthPotion), 1);
  ASSERT_GE(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Chest), 1);
  ASSERT_GE(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Goal), 1);
}

/// Test that creating a map without a seed works correctly.
TEST_F(MapFixture, TestMapCreateMapEmptySeed) {
  const auto [create_map_empty_seed_grid, _] = create_map(0);
  ASSERT_NE(create_map_empty_seed_grid, create_map(0).first);
}

/// Test that creating a map with a negative level throws an exception.
TEST_F(MapFixture, TestMapCreateMapNegativeLevel) {
  ASSERT_THROW_MESSAGE(create_map(-1, 5), std::length_error, "Level must be bigger than or equal to 0.")
}
