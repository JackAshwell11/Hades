// Std includes
#include <algorithm>
#include <cmath>

// ----- ENUMS ------------------------------
/// Stores the different types of tiles in the game map.
enum class TileType {
  DebugWall,
  Empty,
  Floor,
  Wall,
  Obstacle,
  Player,
  HealthPotion,
  ArmourPotion,
  HealthBoostPotion,
  ArmourBoostPotion,
  SpeedBoostPotion,
  FireRateBoostPotion
};

// ----- STRUCTURES ------------------------------
/// Stores a map generation constant which can be calculated.
///
/// Parameters
/// ----------
/// base_value - The base value for the exponential calculation.
/// increase - The percentage increase for the constant.
/// max_value - The max value for the exponential calculation.
struct MapGenerationConstant {
  double base_value, increase, max_value;

  /// Generate a value based on the exponential equation.
  ///
  /// Parameters
  /// ----------
  /// level - The game level to generate a value for.
  ///
  /// Returns
  /// -------
  /// The generated valued.
  inline int generate_value(int level) const {
    return (int)std::min(round(base_value * pow(increase, level)), max_value);
  }
};

/// Stores the map generation constants
///
/// Parameters
/// ----------
/// width - The width of the 2D grid.
/// height - The height of the 2D grid.
/// split_iteration - The amount of splits to perform.
/// obstacle_count - The amount of obstacles to place in the 2D grid.
/// item_count - The amount of items to place in the 2D grid.
struct MapGenerationConstants {
  MapGenerationConstant width, height, split_iteration, obstacle_count,
      item_count;
};

// ----- CONSTANTS ------------------------------
// Defines the constants for the map generation
const MapGenerationConstants MAP_GENERATION_CONSTANTS = {
    {30, 1.2, 150}, {20, 1.2, 100}, {5, 1.5, 25}, {20, 1.3, 200}, {5, 1.1, 30},
};

// Defines the probabilities for each item
const std::pair<TileType, double> ITEM_PROBABILITIES[6] = {
    {TileType::HealthPotion, 0.3},      {TileType::ArmourPotion, 0.3},
    {TileType::HealthBoostPotion, 0.2}, {TileType::ArmourBoostPotion, 0.1},
    {TileType::SpeedBoostPotion, 0.05}, {TileType::FireRateBoostPotion, 0.05},
};

// Defines constants for the binary space partition
const double CONTAINER_RATIO = 1.25;
const int MIN_CONTAINER_SIZE = 5;
const int MIN_ROOM_SIZE = 4;
const double ROOM_RATIO = 0.625;

// Defines constants for hallway and entity generation
const TileType REPLACEABLE_TILES[3] = {TileType::Empty, TileType::Obstacle,
                                       TileType::DebugWall};
const int HALLWAY_SIZE = 5;
