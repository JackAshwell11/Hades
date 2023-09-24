// External includes
#include "gtest/gtest.h"

// Custom includes
#include "generation/bsp.hpp"
#include "generation/primitives.hpp"

// ----- FIXTURES ------------------------------
class GenerationFixtures : public testing::Test {
  /// Hold fixtures relating to the generation/ C++ tests.
 protected:
  Position valid_position_one{3, 5}, valid_position_two{5, 7}, boundary_position{4, 0}, zero_position{0, 0};
  Rect valid_rect_one{valid_position_one, valid_position_two}, valid_rect_two{valid_position_one, boundary_position},
      zero_size_rect{zero_position, zero_position};
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
