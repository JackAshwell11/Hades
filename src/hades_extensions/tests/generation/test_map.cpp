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

  /// An empty map generator for use in testing.
  MapGenerator empty_map{-50, random_generator};  // 0x0 grid

  /// A small map generator for use in testing.
  MapGenerator small_map{-5, random_generator};  // 12x8 grid

  /// A large map generator for use in testing.
  MapGenerator large_map{0, random_generator};  // 30x20 grid

  /// A rect that fits inside the grid for use in testing.
  const Rect rect_one{{.x = 0, .y = 3}, {.x = 3, .y = 7}};

  /// A rect that fits inside the grid for use in testing.
  const Rect rect_two{{.x = 6, .y = 1}, {.x = 11, .y = 5}};

  /// A large rect that doesn't fit inside the grid for use in testing.
  const Rect rect_three{{.x = 14, .y = 1}, {.x = 28, .y = 18}};

  /// Set up the fixture for the tests.
  void SetUp() override { random_generator.seed(0); }

  // /// Place a rect made up of walls and floors in the grid for use in testing.
  // void place_covered_box() const {
  //   very_large_grid.place_rect({{.x = 2, .y = 2}, {.x = 20, .y = 20}});
  //   for (int y = 1; y <= 21; y++) {
  //     for (int x = 1; x <= 21; x++) {
  //       if (very_large_grid.get_value({.x = x, .y = y}) != TileType::Floor) {
  //         very_large_grid.set_value({.x = x, .y = y}, TileType::Wall);
  //       }
  //     }
  //   }
  // }
};

namespace {
/// Assert that there are no adjacent walls to the specified tile type in the grid.
///
/// @param grid - The grid to check for adjacent walls.
/// @param tile_type - The tile type to check for adjacent walls.
void assert_no_adjacent_walls(const Grid &grid, const TileType tile_type) {
  for (int i{0}; i < grid.width * grid.height; i++) {
    if (const Position pos{grid.convert_position(i)}; grid.get_value(pos) == tile_type) {
      const auto neighbours{grid.get_neighbours(pos)};
      ASSERT_EQ(std::ranges::count_if(neighbours.begin(), neighbours.end(),
                                      [&grid](const auto &pos) { return grid.get_value(pos) == TileType::Wall; }),
                0);
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
  for (int i{0}; i < grid.width * grid.height; i++) {
    if (const Position pos{grid.convert_position(i)}; grid.get_value(pos) == tile_type) {
      positions.push_back(pos);
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

/// Test that generating rooms in an empty map doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorGenerateRoomsEmptyMap) {
  const auto rooms{empty_map.generate_rooms().get_rooms()};
  ASSERT_EQ(rooms.size(), 0);
}

/// Test that generating rooms in a small map works correctly.
TEST_F(MapFixture, TestMapGeneratorGenerateRoomsSmallMap) {
  const auto rooms{small_map.generate_rooms().get_rooms()};
  ASSERT_GE(rooms.size(), 2);
  ASSERT_LE(rooms.size(), 2);
}

/// Test that generating rooms in a large map works correctly.
TEST_F(MapFixture, TestMapGeneratorGenerateRoomsLargeMap) {
  const auto rooms{large_map.generate_rooms().get_rooms()};
  ASSERT_GE(rooms.size(), 12);
  ASSERT_LE(rooms.size(), 12);
}

/// Test that generating the connections with a single room doesn't do anything.
TEST_F(MapFixture, TestMapCreateConnectionsSingleRoom) {
  small_map.get_rooms().emplace_back(rect_one.centre);
  const auto connections{small_map.create_connections().get_connections()};
  ASSERT_EQ(connections.size(), 0);
}

/// Test that generating the connections with multiple rooms works correctly.
TEST_F(MapFixture, TestMapCreateConnectionsMultipleRooms) {
  // Create the minimum-spanning tree and check its size
  large_map.get_rooms().emplace_back(rect_one.centre);
  large_map.get_rooms().emplace_back(rect_two.centre);
  large_map.get_rooms().emplace_back(rect_three.centre);
  const auto connections{large_map.create_connections().get_connections()};
  ASSERT_EQ(connections.size(), 2);

  // Check that the minimum spanning tree has the correct total cost
  ASSERT_EQ(std::accumulate(connections.begin(), connections.end(), 0,
                            [](const int &sum, const Connection &connection) { return sum + connection.cost; }),
            19);

  // Check that all the connections are correct
  const std::vector<Connection> expected_connections{
      {.cost = 7, .source = {.x = 2, .y = 5}, .destination = {.x = 9, .y = 3}},
      {.cost = 12, .source = {.x = 9, .y = 3}, .destination = {.x = 21, .y = 10}}};
  for (const auto &connection : connections) {
    ASSERT_TRUE(std::ranges::any_of(connections.begin(), connections.end(),
                                    [&connection](const Connection &other) { return connection == other; }));
  }
}

/// Test that generating the connections with zero rooms throws an exception.
TEST_F(MapFixture, TestMapCreateConnectionsEmptyRooms){
    ASSERT_THROW_MESSAGE(small_map.create_connections(), std::length_error, "Rooms size must be bigger than 0.")}

/// Test that generating hallways with a single connection works correctly.
TEST_F(MapFixture, TestMapGenerateHallwaysSingleConnection) {
  small_map.get_connections().emplace_back(
      Connection{.cost = 0, .source = rect_one.centre, .destination = rect_two.centre});
  small_map.generate_hallways();
  const std::vector single_connection_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*small_map.get_grid().grid, single_connection_result);
}

/// Test that generating hallways with no connections doesn't do anything.
TEST_F(MapFixture, TestMapGenerateHallwaysNoConnections) {
  small_map.generate_hallways();
  const std::vector no_connections_result{96, TileType::Empty};
  ASSERT_EQ(*small_map.get_grid().grid, no_connections_result);
}

/// Test that generating hallways in an empty map doesn't do anything.
TEST_F(MapFixture, TestMapGenerateHallwaysEmptyMap) {
  empty_map.get_connections().emplace_back(
      Connection{.cost = 0, .source = rect_one.centre, .destination = rect_two.centre});
  empty_map.generate_hallways();
  const std::vector empty_map_result{0, TileType::Empty};
  ASSERT_EQ(*empty_map.get_grid().grid, empty_map_result);
}

/// Test that running cellular automata on a grid with mixed floor and wall tiles works correctly.
TEST_F(MapFixture, TestMapGeneratorCellularAutomataMixedFloors) {
  const auto &grid{small_map.get_grid()};
  grid.set_value({.x = 1, .y = 1}, TileType::Floor);
  grid.set_value({.x = 2, .y = 1}, TileType::Floor);
  grid.set_value({.x = 3, .y = 1}, TileType::Floor);
  grid.set_value({.x = 1, .y = 2}, TileType::Floor);
  grid.set_value({.x = 3, .y = 2}, TileType::Floor);
  grid.set_value({.x = 1, .y = 3}, TileType::Floor);
  grid.set_value({.x = 2, .y = 3}, TileType::Floor);
  grid.set_value({.x = 3, .y = 3}, TileType::Floor);
  small_map.cellular_automata();
  const std::vector mixed_floor_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty};
  ASSERT_EQ(*grid.grid, mixed_floor_result);
}

/// Test that running multiple cellular automata simulations works correctly.
TEST_F(MapFixture, TestMapGeneratorCellularAutomataMultipleSimulations) {
  const auto &grid{small_map.get_grid()};
  grid.set_value({.x = 1, .y = 1}, TileType::Floor);
  grid.set_value({.x = 2, .y = 1}, TileType::Floor);
  grid.set_value({.x = 3, .y = 1}, TileType::Floor);
  grid.set_value({.x = 1, .y = 2}, TileType::Floor);
  grid.set_value({.x = 3, .y = 2}, TileType::Floor);
  grid.set_value({.x = 1, .y = 3}, TileType::Floor);
  grid.set_value({.x = 2, .y = 3}, TileType::Floor);
  grid.set_value({.x = 3, .y = 3}, TileType::Floor);
  small_map.cellular_automata(2);
  const std::vector multiple_simulation_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
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
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, multiple_simulation_result);
}

/// Test that running cellular automata on an empty grid doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorCellularAutomataEmptyGrid) {
  empty_map.cellular_automata();
  ASSERT_EQ(*empty_map.get_grid().grid, std::vector<TileType>{});
}

/// Test that placing walls in an empty grid doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorGenerateWallsEmptyGrid) {
  empty_map.generate_walls();
  ASSERT_EQ(*empty_map.get_grid().grid, std::vector<TileType>{});
}

/// Test that placing walls in a grid with all empty tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorGenerateWallsAllEmpty) {
  const auto &grid{empty_map.get_grid()};
  empty_map.generate_walls();
  ASSERT_EQ(*grid.grid, std::vector(static_cast<std::size_t>(grid.width * grid.height), TileType::Empty));
}

/// Test that placing walls in a grid with all wall tiles sets all tiles to walls.
TEST_F(MapFixture, TestMapGeneratorGenerateWallsAllWalls) {
  const auto &grid{small_map.get_grid()};
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      grid.set_value({.x = x, .y = y}, TileType::Wall);
    }
  }
  small_map.generate_walls();
  ASSERT_EQ(*grid.grid, std::vector(static_cast<std::size_t>(grid.width * grid.height), TileType::Wall));
}

/// Test that placing walls in a grid with all floor tiles sets the edges to walls.
TEST_F(MapFixture, TestMapGeneratorGenerateWallsAllFloors) {
  const auto &grid{small_map.get_grid()};
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      grid.set_value({.x = x, .y = y}, TileType::Floor);
    }
  }
  small_map.generate_walls();
  const std::vector all_floor_result{
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
  };
  ASSERT_EQ(*grid.grid, all_floor_result);
}

/// Test that placing walls in a grid with a non-uniform floor pattern works correctly.
TEST_F(MapFixture, TestMapGeneratorGenerateWallsMixedFloors) {
  const auto &grid{small_map.get_grid()};
  grid.set_value({.x = 1, .y = 1}, TileType::Floor);
  grid.set_value({.x = 2, .y = 1}, TileType::Floor);
  grid.set_value({.x = 3, .y = 1}, TileType::Floor);
  grid.set_value({.x = 1, .y = 2}, TileType::Floor);
  grid.set_value({.x = 2, .y = 2}, TileType::Floor);
  grid.set_value({.x = 3, .y = 2}, TileType::Floor);
  grid.set_value({.x = 2, .y = 3}, TileType::Floor);
  small_map.generate_walls();
  const std::vector mixed_floor_result{
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Wall,  TileType::Wall,  TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*grid.grid, mixed_floor_result);
}

/// Test that placing obstacles in an empty grid doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlaceObstaclesEmptyGrid) {
  empty_map.place_obstacles();
  ASSERT_EQ(*empty_map.get_grid().grid, std::vector<TileType>{});
}

/// Test that placing obstacles in a grid with all empty tiles works correctly.
TEST_F(MapFixture, TestMapGeneratorPlaceObstaclesAllEmpty) {
  small_map.place_obstacles();
  ASSERT_EQ(
      std::ranges::count(small_map.get_grid().grid->begin(), small_map.get_grid().grid->end(), TileType::Obstacle), 2);
  assert_min_distance(small_map.get_grid(), TileType::Obstacle);
}

/// Test that placing obstacles in a grid with all floor tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlaceObstaclesAllFloors) {
  const auto &grid{small_map.get_grid()};
  for (int i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Floor);
  }
  small_map.place_obstacles();
  ASSERT_EQ(*grid.grid, std::vector(static_cast<std::size_t>(grid.width * grid.height), TileType::Floor));
}

/// Test that placing obstacles in a grid with all wall tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlaceObstaclesAllWalls) {
  const auto &grid{small_map.get_grid()};
  for (int i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Wall);
  }
  small_map.place_obstacles();
  ASSERT_EQ(*grid.grid, std::vector(static_cast<std::size_t>(grid.width * grid.height), TileType::Wall));
}

/// Test that placing a player in an empty grid doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlacePlayerEmptyGrid) {
  empty_map.place_player();
  ASSERT_EQ(*empty_map.get_grid().grid, std::vector<TileType>{});
}

/// Test that placing a player in a grid with all empty tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlacePlayerAllEmpty) {
  small_map.place_player();
  ASSERT_EQ(std::ranges::count(small_map.get_grid().grid->begin(), small_map.get_grid().grid->end(), TileType::Player),
            0);
}

/// Test that placing a player in a grid with all floor tiles works correctly.
TEST_F(MapFixture, TestMapGeneratorPlacePlayerAllFloors) {
  const auto &grid{small_map.get_grid()};
  for (int i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Floor);
  }
  small_map.place_player();
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Player), 1);
  assert_no_adjacent_walls(grid, TileType::Player);
  assert_min_distance(grid, TileType::Player);
}

/// Test that placing a player in a grid with all wall tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlacePlayerAllWalls) {
  const auto &grid{small_map.get_grid()};
  for (int i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Wall);
  }
  small_map.place_player();
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Player), 0);
}

/// Test that placing items in an empty grid doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlaceItemsEmptyGrid) {
  empty_map.place_items();
  ASSERT_EQ(*empty_map.get_grid().grid, std::vector<TileType>{});
}

/// Test that placing items in a grid with all empty tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlaceItemsAllEmpty) {
  small_map.place_items();
  ASSERT_EQ(
      std::ranges::count(small_map.get_grid().grid->begin(), small_map.get_grid().grid->end(), TileType::HealthPotion),
      0);
}

/// Test that placing items in a grid with all floor tiles works correctly.
TEST_F(MapFixture, TestMapGeneratorPlaceItemsAllFloors) {
  const auto &grid{small_map.get_grid()};
  for (int i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Floor);
  }
  small_map.place_items();
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::HealthPotion), 2);
  assert_no_adjacent_walls(grid, TileType::HealthPotion);
  assert_min_distance(grid, TileType::HealthPotion);
}

/// Test that placing items in a grid with all wall tiles doesn't do anything.
TEST_F(MapFixture, TestMapGeneratorPlaceItemsAllWalls) {
  const auto &grid{small_map.get_grid()};
  for (int i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Wall);
  }
  small_map.place_items();
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::HealthPotion), 0);
}

/// Test that generating a goal on an empty grid throws an exception.
TEST_F(MapFixture, TestMapGeneratorPlaceGoalEmptyGrid){
    ASSERT_THROW_MESSAGE(empty_map.place_goal(), std::out_of_range, "Position not within the grid.")}

/// Test that generating a goal on a grid with all empty tiles throws an exception.
TEST_F(MapFixture, TestMapGeneratorPlaceGoalAllEmpty){
    ASSERT_THROW_MESSAGE(small_map.place_goal(), std::out_of_range, "Position not within the grid.")}

/// Test that generating a goal on a grid with no player tile throws an exception.
TEST_F(MapFixture, TestMapGeneratorPlaceGoalNoPlayer) {
  const auto &grid{small_map.get_grid()};
  for (auto i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Floor);
  }
  ASSERT_THROW_MESSAGE(small_map.place_goal(), std::out_of_range, "Position not within the grid.")
}

/// Test that generating a goal on a grid with a player tile works correctly.
TEST_F(MapFixture, TestMapGeneratorPlaceGoalPlayer) {
  const auto &grid{small_map.get_grid()};
  for (auto i{0}; i < grid.width * grid.height; i++) {
    grid.set_value(grid.convert_position(i), TileType::Floor);
  }
  grid.set_value({.x = 0, .y = 0}, TileType::Player);
  small_map.place_goal();
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Goal), 1);
  assert_min_distance(grid, TileType::Goal);
}

// /// Test that creating a map with a valid level and seed works correctly.
// TEST_F(MapFixture, TestMapGeneratorCreateMapValidLevelSeed) {
//   const auto [create_map_valid_grid, create_map_valid_constants]{create_map(0, 10)};
//   ASSERT_EQ(create_map_valid_constants.width, 30);
//   ASSERT_EQ(create_map_valid_constants.height, 20);
//   ASSERT_EQ(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Player), 1);
//   ASSERT_GE(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::HealthPotion),
//   1); ASSERT_GE(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Chest), 1);
//   ASSERT_GE(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Goal), 1);
// }
//
// /// Test that creating a map without a seed works correctly.
// TEST_F(MapFixture, TestMapGeneratorCreateMapEmptySeed) {
//   const auto [create_map_empty_seed_grid, _]{create_map(0)};
//   ASSERT_NE(create_map_empty_seed_grid, create_map(0).first);
// }
//
// /// Test that creating a map with a negative level throws an exception.
// TEST_F(MapFixture, TestMapGeneratorCreateMapNegativeLevel) {
//   ASSERT_THROW_MESSAGE(create_map(-1, 5), std::length_error, "Level must be bigger than or equal to 0.")
// }
