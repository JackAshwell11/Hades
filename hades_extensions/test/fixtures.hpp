// External includes
#include "gtest/gtest.h"

// Custom includes
#include "bsp.hpp"
#include "primitives.hpp"

// ----- FIXTURES ------------------------------
class Fixtures : public testing::Test {
  /// Hold fixtures relating to the C++ tests.
 protected:
  Point valid_point_one{3, 5}, valid_point_two{5, 7}, boundary_point{4, 0}, zero_point{0, 0};
  Rect valid_rect_one{valid_point_one, valid_point_two}, valid_rect_two{valid_point_one, boundary_point},
      zero_size_rect{zero_point, zero_point};
  Leaf leaf{{{0, 0}, {19, 19}}};
  Grid empty_grid, grid = {20, 20}, small_grid = {6, 9}, detailed_grid = {6, 9};
  std::mt19937 random_generator;

  void SetUp() override {
    random_generator.seed(0);
    detailed_grid.grid = std::make_unique<std::vector<TileType>>(std::vector<TileType>{
        TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
        TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle,
        TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty,
        TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Obstacle, TileType::Obstacle, TileType::Empty,
        TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty,
        TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Obstacle,
        TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Obstacle, TileType::Empty,
        TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty,
        TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
    });
  }
};
