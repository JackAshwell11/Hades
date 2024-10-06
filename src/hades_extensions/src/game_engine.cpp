// Related header
#include "game_engine.hpp"

// Local headers
#include "ecs/systems/armour_regen.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/upgrade.hpp"
#include "factories.hpp"

namespace {
// The number of cellular automata runs to perform.
constexpr int CELLULAR_AUTOMATA_SIMULATIONS{3};

// The distance from the player to generate an enemy.
constexpr int ENEMY_GENERATION_DISTANCE{5};

// How many times an enemy should be attempted to be generated.
constexpr int ENEMY_RETRY_ATTEMPTS{3};
}  // namespace

GameEngine::GameEngine(const int level, const std::optional<unsigned int> seed)
    : registry_(std::make_shared<Registry>()), generator_(0, std::mt19937{std::random_device{}()}) {
  if (level < 0) {
    throw std::length_error("Level must be bigger than or equal to 0.");
  }
  const std::mt19937 random_generator{seed.has_value() ? seed.value() : std::random_device{}()};
  generator_ = MapGenerator{level, random_generator};
  generator_.generate_rooms()
      .place_obstacles()
      .create_connections()
      .generate_hallways()
      .cellular_automata(CELLULAR_AUTOMATA_SIMULATIONS)
      .generate_walls()
      .place_player()
      .place_items()
      .place_goal();

  // Add the systems to the registry
  registry_->add_system<ArmourRegenSystem>();
  registry_->add_system<AttackSystem>();
  registry_->add_system<DamageSystem>();
  registry_->add_system<EffectSystem>();
  registry_->add_system<FootprintSystem>();
  registry_->add_system<InventorySystem>();
  registry_->add_system<KeyboardMovementSystem>();
  registry_->add_system<PhysicsSystem>();
  registry_->add_system<SteeringMovementSystem>();
  registry_->add_system<UpgradeSystem>();
}

void GameEngine::create_game_objects() {
  // Create the registry and get the grid and factories
  const auto &grid{*generator_.get_grid().grid};
  const auto &factories{get_factories()};

  // Create the game objects ignoring empty and obstacle tiles
  for (auto i{0}; i < static_cast<int>(grid.size()); i++) {
    const auto tile_type{grid[i]};
    if (tile_type == TileType::Empty || tile_type == TileType::Obstacle) {
      continue;
    }

    // If the tile is not a wall tile, we want an extra floor tile placed at
    // the same position
    static const std::unordered_map<TileType, GameObjectType> tile_to_game_object_type{
        {TileType::Floor, GameObjectType::Floor},
        {TileType::Wall, GameObjectType::Wall},
        {TileType::Goal, GameObjectType::Goal},
        {TileType::Player, GameObjectType::Player},
        {TileType::HealthPotion, GameObjectType::HealthPotion},
        {TileType::Chest, GameObjectType::Chest},
    };
    const auto game_object_type{tile_to_game_object_type.at(tile_type)};
    const auto [x, y]{generator_.get_grid().convert_position(i)};
    if (tile_type != TileType::Wall) {
      registry_->create_game_object(GameObjectType::Floor, cpv(x, y), factories.at(GameObjectType::Floor)());
    }
    registry_->create_game_object(game_object_type, cpv(x, y), factories.at(game_object_type)());
  }
}
