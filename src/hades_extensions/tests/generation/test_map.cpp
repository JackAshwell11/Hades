// Std headers
#include <algorithm>
#include <numeric>

// Local headers
#include "generation/map.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
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
  const Rect rect_one{{0, 1}, {3, 4}};

  /// An extra rect that fits inside the grid for use in testing.
  const Rect rect_two{{2, 1}, {4, 2}};

  /// A large rect that doesn't fit inside the grid for use in testing.
  const Rect rect_three{{4, 4}, {6, 6}};

  /// Set up the fixture for the tests.
  void SetUp() override { random_generator.seed(0); }

  /// Add item and floor tiles to the grid for use in testing.
  ///
  /// @param items The positions of the items to add.
  void add_items_and_floors(const std::unordered_set<Position> &items) const {
    for (int y = 0; y < very_large_grid.height; y++) {
      for (int x = 0; x < very_large_grid.width; x++) {
        very_large_grid.set_value({x, y}, !items.contains({x, y}) ? TileType::Floor : TileType::Obstacle);
      }
    }
  }
};

// ----- TESTS ------------------------------
/// Test that placing a tile randomly in the grid with a count of 0 doesn't do anything.
TEST_F(MapFixture, TestMapPlaceRandomTilesZeroCount) {
  ASSERT_EQ(place_random_tiles(grid, random_generator, TileType::Empty, TileType::Obstacle, 0),
            std::unordered_set<Position>{});
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 0);
}

/// Test that placing a tile randomly in the grid with a count of 1 works correctly.
TEST_F(MapFixture, TestMapPlaceRandomTilesSingleCount) {
  const std::unordered_set<Position> single_count_result{{3, 2}};
  ASSERT_EQ(place_random_tiles(grid, random_generator, TileType::Empty, TileType::Obstacle, 1), single_count_result);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 1);
}

/// Test that placing a tile randomly in the grid with a count of 3 works correctly.
TEST_F(MapFixture, TestMapPlaceRandomTilesMultipleCount) {
  const std::unordered_set<Position> multiple_count_result{{3, 2}, {4, 2}, {1, 3}};
  ASSERT_EQ(place_random_tiles(grid, random_generator, TileType::Empty, TileType::Obstacle, 3), multiple_count_result);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 3);
}

/// Test that placing a tile randomly in the grid with no available positions throws an exception.
TEST_F(MapFixture, TestMapPlaceRandomTilesNoAvailablePositions){
    ASSERT_THROW_MESSAGE(place_random_tiles(grid, random_generator, TileType::Wall, TileType::Obstacle, 1),
                         std::length_error, "Not enough replaceable tiles to place the target tiles.")}

/// Test that placing a tile randomly in an empty grid throws an exception.
TEST_F(MapFixture, TestMapPlaceRandomTilesEmptyGrid){
    ASSERT_THROW_MESSAGE(place_random_tiles({0, 0}, random_generator, TileType::Empty, TileType::Obstacle, 1),
                         std::length_error, "Not enough replaceable tiles to place the target tiles.")}

/// Test that placing a tile using the Dijkstra map with a count of 0 doesn't do anything.
TEST_F(MapFixture, TestMapPlaceDijkstraTilesZeroCount) {
  std::unordered_set<Position> item_positions{};
  add_items_and_floors(item_positions);
  place_dijkstra_tiles(very_large_grid, random_generator, item_positions, TileType::Obstacle, 0);
  ASSERT_EQ(item_positions, std::unordered_set<Position>{});
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Obstacle), 0);
}

/// Test that placing a tile using the Dijkstra map with a count of 1 works correctly.
TEST_F(MapFixture, TestMapPlaceDijkstraTilesSingleCount) {
  std::unordered_set<Position> item_positions{{2, 2}};
  add_items_and_floors(item_positions);
  place_dijkstra_tiles(very_large_grid, random_generator, item_positions, TileType::Obstacle, 1);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Obstacle), 2);
}

/// Test that placing a tile using the Dijkstra map with a count of 3 works correctly.
TEST_F(MapFixture, TestMapPlaceDijkstraTilesMultipleCount) {
  std::unordered_set<Position> item_positions{{2, 2}, {10, 20}};
  add_items_and_floors(item_positions);
  place_dijkstra_tiles(very_large_grid, random_generator, item_positions, TileType::Obstacle, 3);
  ASSERT_EQ(std::ranges::count(very_large_grid.grid->begin(), very_large_grid.grid->end(), TileType::Obstacle), 5);
}

/// Test that placing a tile using the Dijkstra map with no floors throws an exception.
TEST_F(MapFixture, TestMapPlaceDijkstraTilesNoFloors) {
  std::unordered_set<Position> item_positions{};
  ASSERT_THROW_MESSAGE(place_dijkstra_tiles(very_large_grid, random_generator, item_positions, TileType::Obstacle, 1),
                       std::out_of_range, "Position not within the grid.")
}

/// Test that placing a tile using the Dijkstra map with no items doesn't do anything.
TEST_F(MapFixture, TestMapPlaceDijkstraTilesNoItems) {
  std::unordered_set<Position> item_positions{};
  add_items_and_floors(item_positions);
  ASSERT_THROW_MESSAGE(place_dijkstra_tiles(very_large_grid, random_generator, item_positions, TileType::Obstacle, 1),
                       std::out_of_range, "Position not within the grid.")
}

/// Test that placing a tile using the Dijkstra map in an empty grid throws an exception.
TEST_F(MapFixture, TestMapPlaceDijkstraTilesEmptyGrid) {
  std::unordered_set<Position> item_positions{};
  ASSERT_THROW_MESSAGE(place_dijkstra_tiles({0, 0}, random_generator, item_positions, TileType::Obstacle, 1),
                       std::length_error, "Grid size must be bigger than 0.")
}

/// Test that creating a complete graph with a single room works correctly.
TEST_F(MapFixture, TestMapCreateCompleteGraphSingleRoom) {
  const std::unordered_map<Rect, std::vector<Rect>> single_room_result{{rect_one, std::vector<Rect>{}}};
  ASSERT_EQ(create_complete_graph({rect_one}), single_room_result);
}

/// Test that creating a complete graph with multiple rooms works correctly.
TEST_F(MapFixture, TestMapCreateCompleteGraphMultipleRooms) {
  const std::unordered_map<Rect, std::vector<Rect>> multiple_rooms_result{
      {rect_one, std::vector{rect_two, rect_three}},
      {rect_two, std::vector{rect_one, rect_three}},
      {rect_three, std::vector{rect_one, rect_two}}};
  ASSERT_EQ(create_complete_graph({rect_one, rect_two, rect_three}), multiple_rooms_result);
}

/// Test that creating a complete graph with no rooms throws an exception.
TEST_F(MapFixture, TestMapCreateCompleteGraphNoRooms){
    ASSERT_THROW_MESSAGE(create_complete_graph({}), std::length_error, "Rooms size must be bigger than 0.")}

/// Test that creating a minimum spanning tree with a valid complete graph works correctly.
TEST_F(MapFixture, TestMapCreateConnectionsValidCompleteGraph) {
  // Create the minimum-spanning tree and check its size
  auto connections{create_connections(
      {{rect_one, {rect_two, rect_three}}, {rect_two, {rect_one, rect_three}}, {rect_three, {rect_one, rect_two}}})};
  ASSERT_EQ(connections.size(), 2);

  // Check that the minimum spanning tree has the correct total cost
  ASSERT_EQ(std::accumulate(connections.begin(), connections.end(), 0,
                            [](const int &sum, const Edge &edge) { return sum + edge.cost; }),
            4);

  // Check that every rect can be reached in the minimum spanning tree
  for (const auto &rect : {rect_one, rect_two, rect_three}) {
    ASSERT_TRUE(std::ranges::any_of(connections.begin(), connections.end(), [&rect](const Edge &edge) {
      return edge.source == rect || edge.destination == rect;
    }));
  }
}

/// Test that creating a minimum spanning tree with an empty complete graph throws an exception.
TEST_F(MapFixture, TestMapCreateConnectionsEmptyCompleteGraph){
    ASSERT_THROW_MESSAGE(create_connections({}), std::length_error, "Complete graph size must be bigger than 0.")}

/// Test that creating hallways with a single connection works correctly.
TEST_F(MapFixture, TestMapCreateHallwaysSingleConnection) {
  create_hallways(large_grid, {{0, rect_one, rect_three}});
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
  create_hallways(large_grid, {{0, rect_one, rect_two}, {0, rect_one, rect_three}});
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
  ASSERT_EQ(*grid.grid, std::vector(grid.width * grid.height, TileType::Empty));
}

/// Test that running cellular automata on a grid with all floor tiles sets the edges to walls.
TEST_F(MapFixture, TestMapRunCellularAutomataAllFloors) {
  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      grid.set_value({x, y}, TileType::Floor);
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
      grid.set_value({x, y}, TileType::Wall);
    }
  }
  run_cellular_automata(grid);
  ASSERT_EQ(*grid.grid, std::vector(grid.width * grid.height, TileType::Empty));
}

/// Test that running cellular automata on a grid with mixed floor and wall tiles works correctly.
TEST_F(MapFixture, TestMapRunCellularAutomataMixedFloors) {
  grid.set_value({1, 1}, TileType::Floor);
  grid.set_value({2, 1}, TileType::Floor);
  grid.set_value({3, 1}, TileType::Floor);
  grid.set_value({1, 2}, TileType::Floor);
  grid.set_value({3, 2}, TileType::Floor);
  grid.set_value({1, 3}, TileType::Floor);
  grid.set_value({2, 3}, TileType::Floor);
  grid.set_value({3, 3}, TileType::Floor);
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
  grid.set_value({1, 0}, TileType::Floor);
  grid.set_value({2, 0}, TileType::Floor);
  grid.set_value({3, 0}, TileType::Floor);
  grid.set_value({0, 1}, TileType::Floor);
  grid.set_value({0, 2}, TileType::Floor);
  grid.set_value({0, 3}, TileType::Floor);
  grid.set_value({4, 1}, TileType::Floor);
  grid.set_value({4, 2}, TileType::Floor);
  grid.set_value({4, 3}, TileType::Floor);
  grid.set_value({1, 4}, TileType::Floor);
  grid.set_value({2, 4}, TileType::Floor);
  grid.set_value({3, 4}, TileType::Floor);
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
  grid.set_value({0, 0}, TileType::Floor);
  grid.set_value({2, 0}, TileType::Floor);
  grid.set_value({4, 0}, TileType::Floor);
  grid.set_value({0, 2}, TileType::Floor);
  grid.set_value({2, 1}, TileType::Floor);
  grid.set_value({1, 2}, TileType::Floor);
  grid.set_value({2, 2}, TileType::Floor);
  grid.set_value({3, 2}, TileType::Floor);
  grid.set_value({4, 2}, TileType::Floor);
  grid.set_value({2, 3}, TileType::Floor);
  grid.set_value({0, 4}, TileType::Floor);
  grid.set_value({2, 4}, TileType::Floor);
  grid.set_value({4, 4}, TileType::Floor);
  grid.set_value({4, 4}, TileType::Floor);
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
  const auto [create_map_valid_grid, create_map_valid_constants] = create_map(0, 5);
  ASSERT_EQ(create_map_valid_constants.level, 0);
  ASSERT_EQ(create_map_valid_constants.width, 30);
  ASSERT_EQ(create_map_valid_constants.height, 20);
  ASSERT_EQ(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Player), 1);
  ASSERT_EQ(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Potion), 5);
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
