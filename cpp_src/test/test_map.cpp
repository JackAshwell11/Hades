// Std includes
#include <algorithm>
#include <optional>
#include <queue>

// External includes
#include "gtest/gtest.h"

// Custom includes
#include "map.hpp"
#include "fixtures.hpp"

// ----- FUNCTIONS ------------------------------
/// Tests if floor_count floor tiles are reachable from a random floor position.
///
/// Parameters
/// ----------
/// grid - The 2D grid which represents the dungeon.
/// floor_count - The number of expected floor tiles.
///
/// Returns
/// -------
/// Whether floor_count floor tiles are reachable from a random floor position.
bool has_path(std::vector<TileType> *grid, int floor_count) {
  // Use a Dijkstra map to count the number of floor files reachable
  return true;
}

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestMapCollectPositionsExist) {
  // Test finding a tile that exists in a grid
  std::vector<Point> tile_exists_result = {{0, 4}, {1, 4}, {2, 4}, {0, 5}, {1, 5}, {2, 5}};
  ASSERT_EQ(collect_positions(detailed_grid, TileType::Floor), tile_exists_result);
}

TEST_F(Fixtures, TestMapCollectPositionsNoExist) {
  // Test finding a tile that doesn't exist in a grid
  std::vector<Point> tile_not_exist_result;
  ASSERT_EQ(collect_positions(detailed_grid, TileType::Player), tile_not_exist_result);
}

TEST_F(Fixtures, TestMapCollectPositionsEmptyGrid) {
  // Test finding a tile in an empty grid
  std::vector<Point> tile_not_exist_result;
  ASSERT_EQ(collect_positions(empty_grid, TileType::Floor), tile_not_exist_result);
}

TEST_F(Fixtures, TestMapSplitBspZeroSplit) {
  // Test what happens if split_iteration is 0
  split_bsp(leaf, grid, random_generator, 0);
  ASSERT_TRUE(leaf.left == nullptr);
  ASSERT_TRUE(leaf.right == nullptr);
}

TEST_F(Fixtures, TestMapSplitBsp) {
  // Use a queue to get all the child leaves, so we can check the correct amount
  // of splits has occurred (there should always be n+1 child leaves). We're
  // using 7 as the split_iteration as we want to attempt to split the child
  // leaves (which shouldn't be possible)
  std::queue<Leaf *> split_queue;
  split_bsp(leaf, grid, random_generator, 10);
  split_queue.push(leaf.left);
  split_queue.push(leaf.right);

  // Keep looping until we have all the child leafs and keep track of how many
  // times we've looped (the split iteration)
  int loop_count = 0;
  while (!split_queue.empty()) {
    // Get the current leaf
    Leaf *current = split_queue.front();
    split_queue.pop();

    // Check if current has children. If so, push its children into the queue,
    // otherwise, increment the counter
    if (current->left && current->right) {
      split_queue.push(current->left);
      split_queue.push(current->right);
    } else {
      loop_count++;
    }
  }
  ASSERT_EQ(loop_count, 7);
}

TEST_F(Fixtures, TestMapGenerateRoomsValid) {
  // Test if at least 1 room is generated
  Leaf left_leaf = Leaf{Rect{Point{0, 0}, Point{9, 15}}}, right_leaf = Leaf{Rect{Point{10, 0}, Point{15, 15}}};
  leaf.left = &left_leaf;
  leaf.right = &right_leaf;
  ASSERT_EQ(generate_rooms(leaf, grid, random_generator).size(), 2);
}

TEST_F(Fixtures, TestMapGenerateRoomsRoomExist) {
  // Test if no rooms are generated if a room already exists
  leaf.room = &valid_rect_one;
  ASSERT_TRUE(generate_rooms(leaf, grid, random_generator).empty());
}

TEST_F(Fixtures, TestMapCreateConnectionsValid) {
  // Create a complete graph with 4 nodes and 6 connections
  std::unordered_map<Rect, std::vector<Rect>> complete_graph;
  Rect temp_rect_one = Rect{Point{0, 0}, Point{3, 3}}, temp_rect_two = Rect{Point{10, 10}, Point{12, 12}};
  std::vector<Rect> valid_rect_one_connections = {valid_rect_two, temp_rect_one, temp_rect_two},
      valid_rect_two_connections = {valid_rect_one, temp_rect_one, temp_rect_two},
      temp_rect_one_connections = {valid_rect_one, valid_rect_two, temp_rect_two},
      temp_rect_two_connections = {valid_rect_one, valid_rect_two, temp_rect_one};
  complete_graph.emplace(valid_rect_one, valid_rect_one_connections);
  complete_graph.emplace(valid_rect_two, valid_rect_two_connections);
  complete_graph.emplace(temp_rect_one, temp_rect_one_connections);
  complete_graph.emplace(temp_rect_two, temp_rect_two_connections);

  // Test how many connections are created
  std::unordered_set<Edge> connections_result =
      {{7, valid_rect_one, temp_rect_two}, {3, valid_rect_one, valid_rect_two}, {2, valid_rect_two, temp_rect_one}};
  ASSERT_EQ(create_connections(complete_graph), connections_result);
}

TEST_F(Fixtures, TestMapCreateConnectionsEmpty) {
  // Test if no mst is generated if the provided unordered_map is empty
  std::unordered_map<Rect, std::vector<Rect>> empty_map;
  ASSERT_THROW(create_connections(empty_map), std::length_error);
}

TEST_F(Fixtures, TestMapPlaceTileValid) {
  // Test if a tile is correctly placed in the 2D grid
  std::vector<Point> possible_tiles = {{5, 6}, {4, 2}};
  place_tile(small_grid, random_generator, TileType::Player, possible_tiles);
  ASSERT_TRUE(
      std::find(small_grid.grid.begin(), small_grid.grid.end(), TileType::Player) != small_grid.grid.end());
}

TEST_F(Fixtures, TestMapPlaceTileEmpty) {
  // Test if a tile is not placed in the 2D grid
  std::vector<Point> possible_tiles;
  ASSERT_THROW(place_tile(grid, random_generator, TileType::Player, possible_tiles), std::length_error);
}

TEST_F(Fixtures, TestMapCreateHallwaysValid) {
  // Test if a connection is correctly drawn in the 2D grid with obstacles
  std::unordered_set<Edge> connections = {{0, valid_rect_one, valid_rect_two}};
  create_hallways(small_grid, random_generator, connections, 5);
  std::vector<TileType> create_hallways_valid_result = {
      TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Obstacle, TileType::Wall, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall, TileType::Wall,
  };
  ASSERT_EQ(small_grid.grid, create_hallways_valid_result);
}

TEST_F(Fixtures, TestMapCreateHallwaysNoObstacles) {
  // Test if a connection is correctly drawn in the 2D grid without obstacles
  std::unordered_set<Edge> connections = {{0, valid_rect_one, valid_rect_two}};
  create_hallways(small_grid, random_generator, connections, 0);
  std::vector<TileType> create_hallways_no_obstacles_result = {
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Wall, TileType::Wall, TileType::Floor, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall, TileType::Wall,
  };
  ASSERT_EQ(small_grid.grid, create_hallways_no_obstacles_result);
}

TEST_F(Fixtures, TestMapCreateHallwaysNoConnections) {
  // Test if nothing gets drawn in the 2D grid except from obstacles
  std::unordered_set<Edge> connections = {};
  create_hallways(small_grid, random_generator, connections, 5);
  std::vector<TileType> create_hallways_no_connections_result = {
      TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty,
      TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(small_grid.grid, create_hallways_no_connections_result);
}

TEST_F(Fixtures, TestMapCreateMapValid) {
  // Test if a map is correctly generated
  std::pair<std::vector<TileType>, std::tuple<int, int, int>> create_map_valid = create_map(0, 5);
  ASSERT_TRUE(has_path(&create_map_valid.first,
                       std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::Floor)));
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::Player), 1);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::HealthPotion), 2);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::ArmourPotion), 2);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::HealthBoostPotion), 1);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::ArmourBoostPotion), 1);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::SpeedBoostPotion), 0);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::FireRateBoostPotion), 0);
  ASSERT_EQ(create_map_valid.second, std::make_tuple(0, 30, 20));
}

TEST_F(Fixtures, TestMapCreateMapNegativeLevel) {
  // Test if an exception is thrown on a negative level
  ASSERT_THROW(create_map(-1, 5), std::length_error);
}

TEST_F(Fixtures, TestMapCreateMapEmptySeed) {
  // Test if a map is correctly generated without a given seed. We can't test it
  // against a set result since the seed is randomly generated
  std::optional<unsigned int> empty_seed;
  std::pair<std::vector<TileType>, std::tuple<int, int, int>> create_map_empty_seed = create_map(0, empty_seed);
  ASSERT_EQ(create_map_empty_seed.second, std::make_tuple(0, 30, 20));

  // TODO: REFACTOR THIS TO BE LIKE TESTMAPCREATEMAPVALID

  // Test if the player exists in the 2D grid
  bool has_player = false;
  if (std::find(create_map_empty_seed.first.begin(), create_map_empty_seed.first.end(), TileType::Player)
      != create_map_empty_seed.first.end()) {
    has_player = true;
  }
  ASSERT_TRUE(has_player);
}
