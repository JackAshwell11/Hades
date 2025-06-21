// Related header
#include "game_engine.hpp"

// External headers
#include <nlohmann/json.hpp>

// Local headers
#include "ecs/systems/armour_regen.hpp"
#include "ecs/systems/attacks.hpp"
#include "ecs/systems/effects.hpp"
#include "ecs/systems/inventory.hpp"
#include "ecs/systems/level.hpp"
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

/// The number of non-boss levels in the game.
constexpr int LEVEL_COUNT{2};

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
}  // namespace

GameEngine::GameEngine() {
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

auto GameEngine::is_player_touching_type(const GameObjectType game_object_type) const -> bool {
  return get_nearest_item() != -1 && registry_.get_game_object_type(get_nearest_item()) == game_object_type;
}

void GameEngine::set_seed(const unsigned int seed) { game_state_.dungeon_run.random_generator.seed(seed); }

void GameEngine::reset_level(const LevelType level_type) {
  // Always preserve the player for the current dungeon run
  if (get_player_id() != -1) {
    std::unordered_set<GameObjectID> preserved_ids;
    if (level_type != LevelType::Lobby) {
      preserved_ids.insert(get_player_id());
      const auto inventory{registry_.get_component<Inventory>(get_player_id())};
      preserved_ids.insert(inventory->items.begin(), inventory->items.end());
    }
    registry_.clear_game_objects(preserved_ids);
  }

  // Set up the player and game state
  auto &dungeon_run{game_state_.dungeon_run};
  game_state_.current_level = {.floor_positions = {},
                               .is_lobby = (level_type == LevelType::Lobby),
                               .is_boss_level = (level_type == LevelType::Boss)};
  if (level_type == LevelType::Lobby) {
    create_player();
    dungeon_run = {.player_id = get_player_id(),
                   .game_level = registry_.get_component<PlayerLevel>(get_player_id())->level,
                   .level_distribution = {}};
    dungeon_run.level_distribution = std::normal_distribution<>(get_game_level(), LEVEL_DISTRIBUTION_DEVIATION);
  }
  dungeon_run.dungeon_level++;

  // Create the game objects for the level
  Grid grid(0, 0);
  if (level_type == LevelType::Lobby) {
    grid = MapGenerator{}.place_lobby().get_grid();
  } else {
    grid = MapGenerator{get_game_level(), dungeon_run.random_generator}
               .generate_rooms()
               .place_obstacles()
               .create_connections()
               .generate_hallways()
               .cellular_automata(CELLULAR_AUTOMATA_SIMULATIONS)
               .generate_walls()
               .place_player()
               .place_items()
               .place_goal()
               .get_grid();
  }
  create_game_objects(grid, level_type != LevelType::Lobby);

  // Notify the registry of the level reset
  if (level_type == LevelType::Lobby) {
    registry_.notify<EventType::InventoryUpdate>(std::vector<GameObjectID>{});
    registry_.notify<EventType::RangedAttackSwitch>(0);
    registry_.notify<EventType::AttackCooldownUpdate>(get_player_id(), 0.0, 0.0, 0.0);
    registry_.notify<EventType::StatusEffectUpdate>(std::unordered_map<StatusEffectType, double>{});
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
                                                shop_system->get_offering_cost(i, get_player_id()));
  }
}

void GameEngine::on_update(const double delta_time) {
  auto &current_level{game_state_.current_level};
  current_level.nearest_item = registry_.get_system<PhysicsSystem>()->get_nearest_item(get_player_id());
  if (is_player_touching_type(GameObjectType::Goal) && !current_level.is_lobby) {
    if (get_dungeon_level() > LEVEL_COUNT) {
      // Player has completed the game, return to the lobby
      reset_level(LevelType::Lobby);
    } else if (get_dungeon_level() == LEVEL_COUNT) {
      // Player has completed the last level, create a boss level
      reset_level(LevelType::Boss);
    } else {
      // Player has completed a level, reset the level
      reset_level(LevelType::Normal);
    }
    return;
  }
  current_level.enemy_generation_timer += delta_time;
  if (current_level.enemy_generation_timer >= ENEMY_GENERATION_INTERVAL &&
      std::cmp_less(registry_.get_game_object_ids(GameObjectType::Enemy).size(),
                    MapGenerator::get_enemy_limit(get_game_level()))) {
    generate_enemy();
    current_level.enemy_generation_timer = 0.0;
  }
}

void GameEngine::on_fixed_update(const double delta_time) { registry_.update(delta_time); }

void GameEngine::on_key_press(const int symbol, const int /*modifiers*/) const {
  const auto player_movement{registry_.get_component<KeyboardMovement>(get_player_id())};
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
  const auto player_movement{registry_.get_component<KeyboardMovement>(get_player_id())};
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
      registry_.get_system<InventorySystem>()->add_item_to_inventory(get_player_id(), get_nearest_item());
      break;
    case KEY_E:
      if (game_state_.current_level.is_lobby) {
        if (is_player_touching_type(GameObjectType::Goal)) {
          reset_level(LevelType::Normal);
        } else if (is_player_touching_type(GameObjectType::Shop)) {
          registry_.notify<EventType::ShopOpen>();
        }
      } else {
        use_item(get_player_id(), get_nearest_item());
      }
      break;
    case KEY_Z:
      registry_.get_system<AttackSystem>()->previous_ranged_attack(get_player_id());
      break;
    case KEY_X:
      registry_.get_system<AttackSystem>()->next_ranged_attack(get_player_id());
      break;
    default:
      break;
  }
}

auto GameEngine::on_mouse_press(const double /*x*/, const double /*y*/, const int button, const int /*modifiers*/) const
    -> bool {
  if (button == MOUSE_BUTTON_LEFT) {
    return registry_.get_system<AttackSystem>()->do_attack(get_player_id(), AttackType::Ranged);
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

void GameEngine::create_player() {
  game_state_.dungeon_run.player_id =
      registry_.create_game_object(GameObjectType::Player, cpvzero, get_game_object_components(GameObjectType::Player));
}

void GameEngine::create_game_objects(const Grid &grid, const bool store_floor_positions) {
  for (auto i{0}; std::cmp_less(i, grid.grid.size()); i++) {
    const auto game_object_type{grid.grid[i]};
    if (game_object_type == GameObjectType::Empty || game_object_type == GameObjectType::Obstacle) {
      continue;
    }

    // Get the game object's position
    const auto [x, y]{grid.convert_position(i)};
    const auto position{cpv(x, y)};

    // Store the floor position for enemy generation
    if (game_object_type != GameObjectType::Wall) {
      if (store_floor_positions) {
        game_state_.current_level.floor_positions.emplace_back(position);
      }
      if (game_object_type != GameObjectType::Floor) {
        registry_.create_game_object(GameObjectType::Floor, position,
                                     get_game_object_components(GameObjectType::Floor));
      }
    }

    // Handle game object creation
    if (game_object_type == GameObjectType::Player) {
      cpBodySetPosition(*registry_.get_component<KinematicComponent>(get_player_id())->body,
                        grid_pos_to_pixel(position));
    } else {
      registry_.create_game_object(game_object_type, position, get_game_object_components(game_object_type));
    }
  }
}

void GameEngine::generate_enemy() {
  // Get a random floor position and check if it is valid for enemy generation
  const auto &current_level{game_state_.current_level};
  if (current_level.floor_positions.empty()) {
    return;
  }
  auto dist{std::uniform_int_distribution<size_t>(0, current_level.floor_positions.size() - 1)};
  const auto position{current_level.floor_positions.at(dist(game_state_.dungeon_run.random_generator))};
  const bool intersecting_enemies{
      cpSpacePointQueryNearest(registry_.get_space(), position, 0.0,
                               {CP_NO_GROUP, CP_ALL_CATEGORIES, static_cast<cpBitmask>(GameObjectType::Enemy)},
                               nullptr) != nullptr};
  if (const auto player_position{
          cpBodyGetPosition(*registry_.get_component<KinematicComponent>(get_player_id())->body)};
      intersecting_enemies || cpvdist(position, player_position) < ENEMY_GENERATION_DISTANCE) {
    return;
  }

  // Generate the enemy at the position
  const auto enemy_id{
      registry_.create_game_object(GameObjectType::Enemy, position, get_game_object_components(GameObjectType::Enemy))};
  registry_.get_component<SteeringMovement>(enemy_id)->target_id = get_player_id();
}

auto GameEngine::get_game_object_components(const GameObjectType game_object_type)
    -> std::vector<std::shared_ptr<ComponentBase>> {
  auto &dungeon_run{game_state_.dungeon_run};
  return get_factories().at(game_object_type)(
      std::max(0, static_cast<int>(dungeon_run.level_distribution(dungeon_run.random_generator))));
}
