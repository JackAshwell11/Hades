// Related header
#include "game_engine.hpp"

// Local headers
#include "factories.hpp"
#include "generation/bsp.hpp"

/// Stores a map generation constant which can be calculated.
///
/// @param base_value - The base value for the exponential calculation.
/// @param increase - The percentage increase for the constant.
/// @param max_value - The max value for the exponential calculation.
struct MapGenerationConstant {
  /// The base value for the exponential calculation.
  double base_value;

  /// The percentage increase for the constant.
  double increase;

  /// The max value for the exponential calculation.
  double max_value;

  /// Generate a value based on the exponential equation.
  ///
  /// @param level - The game level to generate a value for.
  /// @return The generated value.
  [[nodiscard]] auto generate_value(const int level) const -> int {
    return static_cast<int>(std::min(round(base_value * pow(increase, level)), max_value));
  }
};

/// Holds the constants for a specific level.
struct LevelConstants {
  /// The level of this game.
  int level;

  /// The width of the dungeon.
  int width;

  /// The height of the dungeon.
  int height;

  /// The total number of enemies that should exist in the dungeon.
  int enemy_limit;
};

namespace {
// The number of cellular automata runs to perform.
constexpr int CELLULAR_AUTOMATA_SIMULATIONS{3};

// Function to generate values based on the exponential equation.
constexpr auto generate_value(const double base_value, const double increase, const double max_value, const int level)
    -> int {
  return static_cast<int>(std::min(round(base_value * pow(increase, level)), max_value));
}

// The width of the grid.
constexpr auto WIDTH(const int level) -> int { return generate_value(30, 1.2, 150, level); }

// The height of the grid.
constexpr auto HEIGHT(const int level) -> int { return generate_value(20, 1.2, 100, level); }

// The number of obstacles to place in the grid.
constexpr auto OBSTACLE_COUNT(const int level) -> int { return generate_value(20, 1.3, 200, level); }

// The total number of enemies that should exist for a given level.
constexpr auto ENEMY_LIMIT(const int level) -> int { return generate_value(5, 1.2, 50, level); }

// The chances of placing an item tile in the grid.
constexpr std::array<std::pair<TileType, double>, 2> ITEM_CHANCES{
    {{TileType::HealthPotion, 0.75}, {TileType::Chest, 0.25}}};

// The mapping of tile types to game object types.
const std::unordered_map<TileType, GameObjectType> tile_to_game_object_type{
    {TileType::Wall, GameObjectType::Wall},
    {TileType::Player, GameObjectType::Player},
    {TileType::HealthPotion, GameObjectType::HealthPotion},
    {TileType::Chest, GameObjectType::Chest},
};
}  // namespace

GameEngine::GameEngine(const int level, const std::optional<unsigned int> seed) {
  // If a seed was provided then use it, otherwise generate a new one
  std::mt19937 random_generator{seed.has_value() ? seed.value() : std::random_device{}()};

  // The constants for the current level.
  const LevelConstants constants{
      .level = level, .width = WIDTH(level), .height = HEIGHT(level), .enemy_limit = ENEMY_LIMIT(level)};
  Grid grid{constants.width, constants.height};
  Leaf bsp{{{.x = 0, .y = 0}, {.x = constants.width - 1, .y = constants.height - 1}}};

  // Create the rooms inside the grid using the BSP
  bsp.split(random_generator);
  const auto rooms{bsp.create_room(grid, random_generator)};

  // Place random obstacles in the grid and create the hallways between the rooms
  place_tiles(grid, random_generator, TileType::Obstacle, 1, OBSTACLE_COUNT(level));
  create_hallways(grid, create_connections(rooms));

  // Run some cellular automata simulations on the grid and place the walls around the floor tiles
  for (int _{0}; _ < CELLULAR_AUTOMATA_SIMULATIONS; _++) {
    run_cellular_automata(grid);
  }

  // Place the player as well as the item tiles in the grid
  place_tiles(grid, random_generator, TileType::Player, 1, 1);
  for (const auto &[tile, probability] : ITEM_CHANCES) {
    place_tiles(grid, random_generator, tile, probability);
  }

  // Initialise game objects for each tile in the grid
  for (int y{0}; y < grid.height; y++) {
    for (int x{0}; x < grid.width; x++) {
      if (const auto tile_type{grid.get_value({.x = x, .y = y})}; tile_to_game_object_type.contains(tile_type)) {
        const auto game_object_type{tile_to_game_object_type.at(tile_type)};
        game_object_ids_.emplace_back(registry_->create_game_object(game_object_type, cpv(x, y), get_factories().at(game_object_type)()), game_object_type);
      }
    }
  }
}
