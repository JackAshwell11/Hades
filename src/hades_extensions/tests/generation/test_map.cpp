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
  Grid large_grid{8, 8};

  /// A rect that fits inside the grid for use in testing.
  const Rect rect_one{{0, 1}, {3, 4}};

  /// A rect that fits inside the grid for use in testing.
  const Rect rect_two{{2, 1}, {4, 2}};

  /// A large rect that doesn't fit inside the grid for use in testing.
  const Rect rect_three{{4, 4}, {6, 6}};

  /// Set up the fixture for the tests.
  void SetUp() override { random_generator.seed(0); }
};

// ----- TESTS ------------------------------
/// Test that finding a tile that exists in the grid returns a vector of positions.
TEST_F(MapFixture, TestMapCollectPositionsExist) {
  const std::vector<Position> tile_exists_result{{3, 0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 4}};
  for (const Position &position : tile_exists_result) {
    grid.set_value(position, TileType::Floor);
  }
  ASSERT_EQ(collect_positions(grid, TileType::Floor), tile_exists_result);
}

/// Test that finding a tile that doesn't exist in the grid returns an empty vector.
TEST_F(MapFixture, TestMapCollectPositionsNoExist) { ASSERT_TRUE(collect_positions(grid, TileType::Player).empty()); }

/// Test that finding a tile in an empty grid returns an empty vector.
TEST_F(MapFixture, TestMapCollectPositionsEmptyGrid) {
  const Grid empty_grid{0, 0};
  ASSERT_TRUE(collect_positions(empty_grid, TileType::Floor).empty());
}

/// Test that placing a tile in the grid with available positions works correctly.
TEST_F(MapFixture, TestMapPlaceTileGivenPositions) {
  std::vector<Position> possible_tiles{{5, 6}, {4, 2}};
  place_tile(grid, random_generator, TileType::Player, possible_tiles);
  ASSERT_EQ(std::ranges::count(grid.grid->begin(), grid.grid->end(), TileType::Player), 1);
}

/// Test that placing a tile in the grid with no available positions throws an exception.
TEST_F(MapFixture, TestMapPlaceTileEmpty) {
  std::vector<Position> possible_tiles;
  ASSERT_THROW_MESSAGE(place_tile(grid, random_generator, TileType::Player, possible_tiles), std::length_error,
                       "Possible tiles size must be bigger than 0.")
}

/// Test that creating a complete graph with a single room works correctly.
TEST_F(MapFixture, TestMapCreateCompleteGraphSingleRoom) {
  const std::vector rooms{rect_one};
  const std::unordered_map<Rect, std::vector<Rect>> single_room_result{{rect_one, std::vector<Rect>{}}};
  ASSERT_EQ(create_complete_graph(rooms), single_room_result);
}

/// Test that creating a complete graph with multiple rooms works correctly.
TEST_F(MapFixture, TestMapCreateCompleteGraphMultipleRooms) {
  const std::vector rooms{rect_one, rect_two, rect_three};
  const std::unordered_map<Rect, std::vector<Rect>> multiple_rooms_result{
      {rect_one, std::vector{rect_two, rect_three}},
      {rect_two, std::vector{rect_one, rect_three}},
      {rect_three, std::vector{rect_one, rect_two}}};
  ASSERT_EQ(create_complete_graph(rooms), multiple_rooms_result);
}

/// Test that creating a complete graph with no rooms throws an exception.
TEST_F(MapFixture, TestMapCreateCompleteGraphNoRooms) {
  const std::vector<Rect> rooms;
  ASSERT_THROW_MESSAGE(create_complete_graph(rooms), std::length_error, "Rooms size must be bigger than 0.")
}

/// Test that creating a minimum spanning tree with a valid complete graph works correctly.
TEST_F(MapFixture, TestMapCreateConnectionsValidCompleteGraph) {
  // Create the minimum-spanning tree and check its size
  const std::unordered_map<Rect, std::vector<Rect>> complete_graph{{rect_one, std::vector{rect_two, rect_three}},
                                                                   {rect_two, std::vector{rect_one, rect_three}},
                                                                   {rect_three, std::vector{rect_one, rect_two}}};
  auto connections{create_connections(complete_graph)};
  const std::unordered_set all_rects{rect_one, rect_two, rect_three};
  ASSERT_EQ(connections.size(), 2);

  // Check that the minimum spanning tree has the correct total cost
  ASSERT_EQ(std::accumulate(connections.begin(), connections.end(), 0,
                            [](const int &sum, const Edge &edge) { return sum + edge.cost; }),
            4);

  // Check that every rect can be reached in the minimum spanning tree
  for (const auto &rect : all_rects) {
    ASSERT_TRUE(std::ranges::any_of(connections.begin(), connections.end(), [&rect](const Edge &edge) {
      return edge.source == rect || edge.destination == rect;
    }));
  }
}

/// Test that creating a minimum spanning tree with an empty complete graph throws an exception.
TEST_F(MapFixture, TestMapCreateConnectionsEmptyCompleteGraph) {
  const std::unordered_map<Rect, std::vector<Rect>> empty_complete_graph;
  ASSERT_THROW_MESSAGE(create_connections(empty_complete_graph), std::length_error,
                       "Complete graph size must be bigger than 0.")
}

/// Test that creating hallways with no obstacles works correctly.
TEST_F(MapFixture, TestMapCreateHallwaysNoObstacles) {
  const std::unordered_set<Edge> connections{{0, rect_one, rect_three}};
  create_hallways(large_grid, random_generator, connections, 0);
  const std::vector no_obstacles_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Empty, TileType::Empty, TileType::Wall,  TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Wall,  TileType::Empty,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Wall,  TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
  };
  ASSERT_EQ(*large_grid.grid, no_obstacles_result);
}

/// Test that creating hallways with obstacles works correctly.
TEST_F(MapFixture, TestMapCreateHallwaysObstacles) {
  const std::unordered_set<Edge> connections{{0, rect_one, rect_three}};
  create_hallways(large_grid, random_generator, connections, 5);
  const std::vector obstacles_result{
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall,  TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,  TileType::Wall,  TileType::Floor,
      TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Floor, TileType::Floor, TileType::Floor, TileType::Floor,
      TileType::Floor, TileType::Wall,  TileType::Empty, TileType::Wall,  TileType::Wall,  TileType::Wall,
      TileType::Wall,  TileType::Wall,  TileType::Wall,  TileType::Wall,
  };
  ASSERT_EQ(*large_grid.grid, obstacles_result);
}

/// Test that creating hallways with no connections doesn't do anything.
TEST_F(MapFixture, TestMapCreateHallwaysNoConnections) {
  const std::unordered_set<Edge> connections;
  create_hallways(large_grid, random_generator, connections, 5);
  const std::vector no_obstacles_result{
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Obstacle, TileType::Obstacle, TileType::Obstacle, TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Obstacle,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Obstacle, TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty, TileType::Empty,
      TileType::Empty,    TileType::Empty,    TileType::Empty,    TileType::Empty,
  };
  ASSERT_EQ(*large_grid.grid, no_obstacles_result);
}

/// Test that creating a map with a valid level and seed works correctly.
TEST_F(MapFixture, TestMapCreateMapValidLevelSeed) {
  const auto [create_map_valid_grid, create_map_valid_constants] = create_map(0, 5);
  ASSERT_EQ(create_map_valid_constants, std::make_tuple(0, 30, 20));
  ASSERT_EQ(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Player), 1);
  ASSERT_EQ(std::ranges::count(create_map_valid_grid.begin(), create_map_valid_grid.end(), TileType::Potion), 5);
}

/// Test that creating a map with a negative level throws an exception.
TEST_F(MapFixture, TestMapCreateMapNegativeLevel){
    ASSERT_THROW_MESSAGE(create_map(-1, 5), std::length_error, "Level must be bigger than or equal to 0.")}

/// Test that creating a map without a seed works correctly.
TEST_F(MapFixture, TestMapCreateMapEmptySeed) {
  const auto [create_map_empty_seed_grid, _] = create_map(0);
  ASSERT_NE(create_map_empty_seed_grid, create_map(0).first);
}
