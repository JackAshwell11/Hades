// Std includes
#include <algorithm>
#include <optional>
#include <queue>

// External includes
#include "gtest/gtest.h"

// Custom includes
#include "map.hpp"
#include "fixtures.hpp"

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

TEST_F(Fixtures, TestMapSplitBspNegativeSplit) {
  // Test what happens if split_iteration is -1
  split_bsp(leaf, grid, random_generator, -1);
  ASSERT_TRUE(leaf.left == nullptr);
  ASSERT_TRUE(leaf.right == nullptr);
}

TEST_F(Fixtures, TestMapSplitBspZeroSplit) {
  // Test what happens if split_iteration is 0
  split_bsp(leaf, grid, random_generator, 0);
  ASSERT_TRUE(leaf.left == nullptr);
  ASSERT_TRUE(leaf.right == nullptr);
}

TEST_F(Fixtures, TestMapSplitBspThreeChildren) {
  // Split the bsp 2 times. This should generate two child leafs under the left
  // child and one child leaf on the right
  split_bsp(leaf, grid, random_generator, 2);

  // Test if the child leafs were generated
  ASSERT_TRUE(leaf.left->left->left == nullptr);
  ASSERT_TRUE(leaf.left->left->right == nullptr);
  ASSERT_TRUE(leaf.left->right->left == nullptr);
  ASSERT_TRUE(leaf.left->right->right == nullptr);
  ASSERT_TRUE(leaf.right->left == nullptr);
  ASSERT_TRUE(leaf.right->right == nullptr);
}

TEST_F(Fixtures, TestMapSplitBspMaxSplit) {
  // Split the bsp with a split iteration of 10. We want to keep splitting the
  // bsp until it is no longer possible so this value should ensure that
  split_bsp(leaf, grid, random_generator, 10);

  // Collect all the child leafs so that we can test them
  std::vector<Leaf *> children;
  std::queue<Leaf *> split_queue;
  split_queue.push(&leaf);
  while (!split_queue.empty()) {
    // Get the current leaf
    Leaf *current = split_queue.front();
    split_queue.pop();

    // Check if current has children. If so, push its children into the queue,
    // otherwise, add the child leaf to the children vector
    if (current->left && current->right) {
      split_queue.push(current->left);
      split_queue.push(current->right);
    } else {
      children.emplace_back(current);
    }
  }

  // Test that each child leaf cannot be split anymore and that there are
  // approximately 4-8 children (this varies based on the toolchain)
  for (Leaf *child : children) {
    ASSERT_FALSE(child->split(grid, random_generator, false));
  }
  ASSERT_TRUE(children.size() >= 4 || children.size() <= 8);
}

TEST_F(Fixtures, TestMapGenerateRoomsSetLeaf) {
  // Test if at least 1 room is generated
  Leaf left_leaf = Leaf{{{0, 0}, {9, 15}}}, right_leaf = Leaf{{{10, 0}, {15, 15}}};
  leaf.left = &left_leaf;
  leaf.right = &right_leaf;
  ASSERT_EQ(generate_rooms(leaf, grid, random_generator).size(), 2);
}

TEST_F(Fixtures, TestMapGenerateRoomsRoomExist) {
  // Test if no rooms are generated if a room already exists
  leaf.room = &valid_rect_one;
  ASSERT_TRUE(generate_rooms(leaf, grid, random_generator).empty());
}

TEST_F(Fixtures, TestMapCreateConnectionsGivenConnections) {
  // Create a complete graph with 4 nodes and 6 connections
  std::unordered_map<Rect, std::vector<Rect>> complete_graph;
  Rect temp_rect_one = {{0, 0}, {3, 3}}, temp_rect_two = {{10, 10}, {12, 12}};
  complete_graph.emplace(valid_rect_one, std::vector<Rect>{valid_rect_two, temp_rect_one, temp_rect_two});
  complete_graph.emplace(valid_rect_two, std::vector<Rect>{valid_rect_one, temp_rect_one, temp_rect_two});
  complete_graph.emplace(temp_rect_one, std::vector<Rect>{valid_rect_one, valid_rect_two, temp_rect_two});
  complete_graph.emplace(temp_rect_two, std::vector<Rect>{valid_rect_one, valid_rect_two, temp_rect_one});
  std::unordered_set<Edge> connections = create_connections(complete_graph);

  // Test if all rects are connected and that there are only 3 connections
  std::unordered_set<Rect> discovered;
  for (Edge connection : connections) {
    discovered.emplace(connection.source);
    discovered.emplace(connection.destination);
  }
  ASSERT_EQ(discovered.size(), 4);
  ASSERT_EQ(connections.size(), 3);
}

TEST_F(Fixtures, TestMapCreateConnectionsEmpty) {
  // Test if no mst is generated if the provided unordered_map is empty
  std::unordered_map<Rect, std::vector<Rect>> empty_map;
  ASSERT_THROW(create_connections(empty_map), std::length_error);
}

TEST_F(Fixtures, TestMapPlaceTileGivenPositions) {
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

TEST_F(Fixtures, TestMapCreateHallwaysWithObstacles) {
  // Test if a connection is correctly drawn in the 2D grid with obstacles
  std::unordered_set<Edge> connections = {{0, valid_rect_one, valid_rect_two}};
  create_hallways(small_grid, random_generator, connections, 5);

  // Get the first floor tile in the grid
  int index =
      (int) (std::find(small_grid.grid.begin(), small_grid.grid.end(), TileType::Floor) - small_grid.grid.begin());
  Point start = {index % small_grid.width, index / small_grid.width};

  // Use a Dijkstra map to count the number of floor tiles reachable
  std::unordered_set<Point> tiles;
  std::deque<Point> queue = {start};
  std::vector<Point> offsets = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
  while (!queue.empty()) {
    // Get the current point to explore
    Point current = queue.front();
    queue.pop_front();

    // Get the current tile's neighbours
    for (Point offset : offsets) {
      // Calculate the neighbour's position and check if its valid excluding the
      // boundaries as that produces weird paths
      Point neighbour = current + offset;
      if (neighbour.x < 0 || neighbour.x >= small_grid.width || neighbour.y < 0 || neighbour.y >= small_grid.height) {
        continue;
      } else if (small_grid.get_value(neighbour) == TileType::Floor && !tiles.contains(neighbour)) {
        queue.push_back(neighbour);
        tiles.emplace(neighbour);
      }
    }
  }

  // Determine if the number of floor tiles generated matches the number of
  // traversable floor tiles
  ASSERT_EQ(std::count(small_grid.grid.begin(), small_grid.grid.end(), TileType::Floor), tiles.size());
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

  // Determine if there's 5 obstacles and no floor tiles
  ASSERT_EQ(std::count(small_grid.grid.begin(), small_grid.grid.end(), TileType::Obstacle), 5);
  ASSERT_EQ(std::count(small_grid.grid.begin(), small_grid.grid.end(), TileType::Floor), 0);
}

TEST_F(Fixtures, TestMapCreateMapCorrect) {
  // Test if a map is correctly generated
  std::pair<std::vector<TileType>, std::tuple<int, int, int>> create_map_valid = create_map(0, 5);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::Player), 1);
  ASSERT_EQ(std::count(create_map_valid.first.begin(), create_map_valid.first.end(), TileType::Potion), 5);
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

  // Test if the player exists in the 2D grid
  bool has_player = false;
  if (std::find(create_map_empty_seed.first.begin(), create_map_empty_seed.first.end(), TileType::Player)
      != create_map_empty_seed.first.end()) {
    has_player = true;
  }
  ASSERT_TRUE(has_player);
}
