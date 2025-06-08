// Related header
#include "game_engine.hpp"

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/systems/armour_regen.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/movements.hpp"
#include "ecs/systems/physics.hpp"
#include "ecs/systems/shop.hpp"
#include "factories.hpp"

namespace {
/// The deviation of the level distribution.
constexpr int LEVEL_DISTRIBUTION_DEVIATION{2};

/// The number of cellular automata runs to perform.
constexpr int CELLULAR_AUTOMATA_SIMULATIONS{3};

/// The distance from the player to generate an enemy.
constexpr double ENEMY_GENERATION_DISTANCE{5 * SPRITE_SIZE};

/// The interval in seconds between enemy generations.
constexpr double ENEMY_GENERATION_INTERVAL{1.0};

/// Get a component type from a string.
///
/// @param type - The string representation of the component type.
/// @throws std::runtime_error if the type is not recognised.
/// @return The type index of the component type.
auto get_component_type_from_string(const std::string &type) -> std::type_index {
  static const std::unordered_map<std::string, std::type_index> type_map{{"Health", typeid(Health)},
                                                                         {"Armour", typeid(Armour)}};
  if (type_map.contains(type)) {
    return type_map.at(type);
  }
  throw std::runtime_error("Unknown component type: " + type);
}

/// Get a mapping from tile types to game object types.
///
/// @return A constant reference to the mapping.
auto get_tile_to_game_object_type() -> const std::unordered_map<TileType, GameObjectType> & {
  static const std::unordered_map<TileType, GameObjectType> mapping{
      {TileType::Floor, GameObjectType::Floor},
      {TileType::Wall, GameObjectType::Wall},
      {TileType::Goal, GameObjectType::Goal},
      {TileType::Player, GameObjectType::Player},
      {TileType::HealthPotion, GameObjectType::HealthPotion},
      {TileType::Chest, GameObjectType::Chest}};
  return mapping;
}
}  // namespace

GameEngine::GameEngine(const int level, const std::optional<unsigned int> seed)
    : random_generator_(seed.has_value() ? seed.value() : std::random_device{}()),
      level_distribution_(level, LEVEL_DISTRIBUTION_DEVIATION),
      level_(level) {
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
  registry_.add_system<ArmourRegenSystem>();
  registry_.add_system<AttackSystem>();
  registry_.add_system<DamageSystem>();
  registry_.add_system<EffectSystem>();
  registry_.add_system<FootprintSystem>();
  registry_.add_system<InventorySystem>();
  registry_.add_system<KeyboardMovementSystem>();
  registry_.add_system<PhysicsSystem>();
  registry_.add_system<ShopSystem>();
  registry_.add_system<SteeringMovementSystem>();
}

void GameEngine::create_game_objects() {
  // Create the game objects ignoring empty and obstacle tiles
  const auto &grid{*generator_.get_grid().grid};
  for (auto i{0}; std::cmp_less(i, grid.size()); i++) {
    const auto tile_type{grid[i]};
    if (tile_type == TileType::Empty || tile_type == TileType::Obstacle) {
      continue;
    }

    // Get the game object's type and position
    const auto game_object_type{get_tile_to_game_object_type().at(tile_type)};
    const auto [x, y]{generator_.get_grid().convert_position(i)};
    const auto position{cpv(x, y)};

    // Store the floor position for enemy generation
    if (tile_type != TileType::Wall) {
      floor_positions_.emplace_back(position);
      if (tile_type != TileType::Floor) {
        registry_.create_game_object(GameObjectType::Floor, position,
                                     get_game_object_components(GameObjectType::Floor));
      }
    }

    // Handle game object creation
    const auto game_object_id{
        registry_.create_game_object(game_object_type, cpv(x, y), get_game_object_components(game_object_type))};
    if (tile_type == TileType::Player) {
      player_id_ = game_object_id;
    }
  }
}

void GameEngine::setup_shop(std::istream &stream) const {
  const auto shop_system{registry_.get_system<ShopSystem>()};
  nlohmann::json offerings;
  stream >> offerings;
  for (int i{0}; std::cmp_less(i, offerings.size()); i++) {
    const auto &offering{offerings[i]};
    const auto type{offering["type"].get<std::string>()};
    const auto name{offering["name"].get<std::string>()};
    const auto description{offering["description"].get<std::string>()};
    const auto icon_type{offering["icon_type"].get<std::string>()};
    const auto base_cost{offering["base_cost"].get<double>()};
    const auto cost_multiplier{offering["cost_multiplier"].get<double>()};
    if (type == "stat") {
      shop_system->add_stat_upgrade(name, description, get_component_type_from_string(offering["stat_type"]), base_cost,
                                    cost_multiplier, offering["base_value"], offering["value_multiplier"]);
    } else if (type == "component") {
      shop_system->add_component_unlock(name, description, base_cost, cost_multiplier);
    } else if (type == "item") {
      shop_system->add_item(name, description, base_cost, cost_multiplier);
    } else {
      throw std::runtime_error("Unknown offering type: " + type);
    }
    registry_.notify<EventType::ShopItemLoaded>(i, std::make_tuple(name, description, icon_type),
                                                shop_system->get_offering_cost(i, player_id_));
  }
}

void GameEngine::on_update(const double delta_time) {
  nearest_item_ = registry_.get_system<PhysicsSystem>()->get_nearest_item(player_id_);
  if (nearest_item_ != -1 && registry_.get_game_object_type(nearest_item_) == GameObjectType::Goal) {
    registry_.delete_game_object(player_id_);
  }
  enemy_generation_timer_ += delta_time;
  if (enemy_generation_timer_ >= ENEMY_GENERATION_INTERVAL) {
    generate_enemy();
    enemy_generation_timer_ = 0.0;
  }
}

void GameEngine::on_fixed_update(const double delta_time) { registry_.update(delta_time); }

void GameEngine::on_key_press(const int symbol, const int /*modifiers*/) const {
  const auto player_movement{registry_.get_component<KeyboardMovement>(player_id_)};
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

void GameEngine::on_key_release(const int symbol, const int /*modifiers*/) {
  const auto player_movement{registry_.get_component<KeyboardMovement>(player_id_)};
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
      registry_.get_system<InventorySystem>()->add_item_to_inventory(player_id_, nearest_item_);
      break;
    case KEY_E:
      use_item(player_id_, nearest_item_);
      break;
    case KEY_Z:
      registry_.get_system<AttackSystem>()->previous_ranged_attack(player_id_);
      break;
    case KEY_X:
      registry_.get_system<AttackSystem>()->next_ranged_attack(player_id_);
      break;
    default:
      break;
  }
}

auto GameEngine::on_mouse_press(const double /*x*/, const double /*y*/, const int button, const int /*modifiers*/) const
    -> bool {
  if (button == MOUSE_BUTTON_LEFT) {
    return registry_.get_system<AttackSystem>()->do_attack(player_id_, AttackType::Ranged);
  }
  return false;
}

void GameEngine::use_item(const GameObjectID target_id, const GameObjectID item_id) {
  // Check if the item is a valid game object or not
  if (!registry_.has_game_object(item_id)) {
    return;
  }

  // Use the item if it can be used
  bool used{false};
  if (registry_.has_component(item_id, typeid(EffectApplier))) {
    used = registry_.get_system<EffectSystem>()->apply_effects(item_id, target_id);
  }

  // If the item has been used, remove it from the inventory or the dungeon
  if (used) {
    if (const auto inventory_system{registry_.get_system<InventorySystem>()};
        inventory_system->has_item_in_inventory(target_id, item_id)) {
      inventory_system->remove_item_from_inventory(target_id, item_id);
    } else {
      registry_.delete_game_object(item_id);
    }
  }
}

void GameEngine::generate_enemy() {
  if (std::cmp_greater_equal(registry_.get_game_object_ids(GameObjectType::Enemy).size(),
                             generator_.get_enemy_limit())) {
    return;
  }

  // Get a random floor position and check if it is valid for enemy generation
  auto dist{std::uniform_int_distribution<size_t>(0, floor_positions_.size() - 1)};
  const auto position{floor_positions_[dist(random_generator_)]};
  const bool intersecting_enemies{
      cpSpacePointQueryNearest(registry_.get_space(), position, 0.0,
                               {CP_NO_GROUP, CP_ALL_CATEGORIES, static_cast<cpBitmask>(GameObjectType::Enemy)},
                               nullptr) != nullptr};
  if (const auto player_position{cpBodyGetPosition(*registry_.get_component<KinematicComponent>(player_id_)->body)};
      intersecting_enemies || cpvdist(position, player_position) < ENEMY_GENERATION_DISTANCE) {
    return;
  }

  // Generate the enemy at the position
  const auto enemy_id{
      registry_.create_game_object(GameObjectType::Enemy, position, get_game_object_components(GameObjectType::Enemy))};
  registry_.get_component<SteeringMovement>(enemy_id)->target_id = player_id_;
}

auto GameEngine::get_game_object_components(const GameObjectType game_object_type)
    -> std::vector<std::shared_ptr<ComponentBase>> {
  return get_factories().at(game_object_type)(std::max(0, static_cast<int>(level_distribution_(random_generator_))));
}
