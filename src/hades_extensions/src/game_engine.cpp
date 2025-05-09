// Related header
#include "game_engine.hpp"

// Std headers
#include <utility>

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
// The deviation of the level distribution.
constexpr int LEVEL_DISTRIBUTION_DEVIATION{2};

// The number of cellular automata runs to perform.
constexpr int CELLULAR_AUTOMATA_SIMULATIONS{3};

// The distance from the player to generate an enemy.
constexpr int ENEMY_GENERATION_DISTANCE{5};

// How many times an enemy should be attempted to be generated.
constexpr int ENEMY_RETRY_ATTEMPTS{3};
}  // namespace

GameEngine::GameEngine(const int level, const std::optional<unsigned int> seed)
    : level_(level),
      random_generator_(seed.has_value() ? seed.value() : std::random_device{}()),
      level_distribution_(level, LEVEL_DISTRIBUTION_DEVIATION),
      registry_(std::make_shared<Registry>(random_generator_)),
      player_id_(-1) {
  if (level < 0) {
    throw std::length_error("Level must be bigger than or equal to 0.");
  }
  generator_ = MapGenerator{level, random_generator_};
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

auto GameEngine::get_level_constants() -> std::tuple<int, int, int> {
  return {generator_.get_grid().width, generator_.get_grid().height, generator_.get_enemy_limit()};
}

void GameEngine::create_game_objects() {
  // Create the game objects ignoring empty and obstacle tiles
  const auto &grid{*generator_.get_grid().grid};
  for (auto i{0}; std::cmp_less(i, grid.size()); i++) {
    const auto tile_type{grid[i]};
    if (tile_type == TileType::Empty || tile_type == TileType::Obstacle) {
      continue;
    }

    // If the tile is not a wall tile, we want an extra floor tile placed at the same position
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
    if (tile_type != TileType::Wall && tile_type != TileType::Floor) {
      registry_->create_game_object(GameObjectType::Floor, cpv(x, y),
                                    get_game_object_components(GameObjectType::Floor));
    }
    const auto game_object_id{
        registry_->create_game_object(game_object_type, cpv(x, y), get_game_object_components(game_object_type))};
    if (tile_type == TileType::Player) {
      player_id_ = game_object_id;
    }
  }
}

void GameEngine::generate_enemy(const double /*delta_time*/) {
  if (static_cast<int>(registry_->get_game_object_ids(GameObjectType::Enemy).size()) >= generator_.get_enemy_limit()) {
    return;
  }

  // Collect all floor positions and shuffle them
  const auto &grid{*generator_.get_grid().grid};
  std::vector<cpVect> floor_positions;
  for (auto i{0}; std::cmp_less(i, grid.size()); i++) {
    if (grid[i] == TileType::Floor) {
      const auto [x, y]{generator_.get_grid().convert_position(i)};
      floor_positions.push_back(cpv(x, y));
    }
  }
  std::ranges::shuffle(floor_positions, random_generator_);

  // Determine which floor to place the enemy on only trying ENEMY_RETRY_ATTEMPTS times
  for (auto attempt{0}; attempt < std::min(static_cast<int>(floor_positions.size()), ENEMY_RETRY_ATTEMPTS); attempt++) {
    const auto position{floor_positions[attempt]};
    if (const auto player_position{cpBodyGetPosition(*registry_->get_component<KinematicComponent>(player_id_)->body)};
        cpSpacePointQueryNearest(registry_->get_space(), position, 0.0,
                                 {CP_NO_GROUP, CP_ALL_CATEGORIES, static_cast<cpBitmask>(GameObjectType::Enemy)},
                                 nullptr) != nullptr ||
        cpvdist(position, player_position) < ENEMY_GENERATION_DISTANCE * SPRITE_SIZE) {
      continue;
    }

    // Create the enemy and set its required data
    const auto enemy_id{registry_->create_game_object(GameObjectType::Enemy, position,
                                                      get_game_object_components(GameObjectType::Enemy))};
    registry_->get_component<SteeringMovement>(enemy_id)->target_id = player_id_;
    return;
  }
}

void GameEngine::on_update(const double /*delta_time*/) {
  nearest_item_ = registry_->get_system<PhysicsSystem>()->get_nearest_item(player_id_);
  if (nearest_item_ != -1 && registry_->get_game_object_type(nearest_item_) == GameObjectType::Goal) {
    registry_->delete_game_object(player_id_);
  }
}

void GameEngine::on_fixed_update(const double delta_time) const { registry_->update(delta_time); }

void GameEngine::on_key_press(const int symbol, const int /*modifiers*/) const {
  const auto player_movement{registry_->get_component<KeyboardMovement>(player_id_)};
  switch (symbol) {
    case KEY_W:
      player_movement->moving_north = true;
      break;
    case KEY_A:
      player_movement->moving_west = true;
      break;
    case KEY_S:
      player_movement->moving_south = true;
      break;
    case KEY_D:
      player_movement->moving_east = true;
      break;
    default:
      break;
  }
}

void GameEngine::on_key_release(const int symbol, const int /*modifiers*/) const {
  const auto player_movement{registry_->get_component<KeyboardMovement>(player_id_)};
  switch (symbol) {
    case KEY_W:
      player_movement->moving_north = false;
      break;
    case KEY_A:
      player_movement->moving_west = false;
      break;
    case KEY_S:
      player_movement->moving_south = false;
      break;
    case KEY_D:
      player_movement->moving_east = false;
      break;
    case KEY_C:
      registry_->get_system<InventorySystem>()->add_item_to_inventory(player_id_, nearest_item_);
      break;
    case KEY_E:
      registry_->get_system<InventorySystem>()->use_item(player_id_, nearest_item_);
      break;
    case KEY_Z:
      registry_->get_component<Attack>(player_id_)->previous_ranged_attack();
      break;
    case KEY_X:
      registry_->get_component<Attack>(player_id_)->next_ranged_attack();
      break;
    default:
      break;
  }
}

auto GameEngine::on_mouse_press(const double /*x*/, const double /*y*/, const int button, const int /*modifiers*/) const
    -> bool {
  if (button == MOUSE_BUTTON_LEFT) {
    return registry_->get_system<AttackSystem>()->do_attack(player_id_, AttackType::Ranged);
  }
  return false;
}

auto GameEngine::get_game_object_components(const GameObjectType game_object_type)
    -> std::vector<std::shared_ptr<ComponentBase>> {
  const auto &factories{get_factories()};
  return factories.at(game_object_type)(std::max(0, static_cast<int>(level_distribution_(random_generator_))));
}
