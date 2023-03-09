// External includes
#include "gtest/gtest.h"

// Custom includes
#include "bsp.hpp"
#include "primitives.hpp"

// ----- FIXTURES ------------------------------
class Fixtures : public testing::Test {
 protected:
  Point valid_point_one{3, 5}, valid_point_two{5, 7}, boundary_point{4, 0}, zero_point{0, 0};
  Rect valid_rect_one{valid_point_one, valid_point_two}, valid_rect_two{valid_point_one, boundary_point},
      zero_size_rect{zero_point, zero_point};
  Leaf leaf{Rect{Point{0, 0}, Point{19, 19}}};
  std::mt19937 random_generator;
  std::vector<std::vector<TileType>> detailed_grid = {
      {TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty},
      {TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Obstacle, TileType::Obstacle, TileType::Empty},
      {TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Floor, TileType::Floor, TileType::Floor, TileType::Empty, TileType::Empty, TileType::Obstacle},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Obstacle, TileType::Empty},
      {TileType::Empty, TileType::Empty, TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Obstacle, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
  };
  std::vector<std::vector<TileType>> grid, small_grid, empty_grid;

  void SetUp() override {
    grid = std::vector<std::vector<TileType>>(20, std::vector<TileType>(20, TileType::Empty));
    small_grid = std::vector<std::vector<TileType>>(9, std::vector<TileType>(6, TileType::Empty));
    random_generator.seed(0);
  }
};
